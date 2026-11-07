"""PIIAnonymizer: turn detected spans into placeholder tokens and masked text.

Steps in `resolve`:
1. Drop spans the anonymizer must not mask: allow-listed values, labels outside `included_entities`,
   and (with `unmask_url`) spans inside a URL.
2. Give each span an identity: spans whose values are the same under the label's variant rules
   (see span_variants.py) share one. Identities are numbered per label in order of first appearance.
3. For labels in `entities_to_look_out_for`, find every other copy of a detected value in the text,
   in any format the variant rules accept, and give it the same identity.
4. Subclass hook `_extra_spans` (OptimizedAnonymizer adds short forms of names here).
5. Merge neighbouring spans of the same identity separated only by spaces or tabs.
6. Assign tokens: one per identity (original mode) or one per exact surface within an identity
   (versioned mode). Surfaces given to `load_mappings` keep their user-provided token.

An instance remembers the identities and tokens it has issued, so one instance used for several
documents keeps tokens consistent across them. PIIPipeline uses a fresh instance per document.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import replace

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ..core.errors import ChecksumMismatchError
from ..core.types import DetectionResult, PIISpan, checksum_of
from ..nlp.context_enhancer import is_inside_ascii_word
from .span_variants import find_variants, identity_key, rules_for
from .tokens import format_token, parse_token

log = logging.getLogger(__name__)

_URL = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'`)\]]+", re.IGNORECASE)
_HSPACE = re.compile(r"[ \t]+")
MIN_PROPAGATED_CHARS = 2


def span_from_result(r: RecognizerResult, text: str) -> PIISpan:
    meta = dict(r.recognition_metadata or {})
    name = meta.pop(RecognizerResult.RECOGNIZER_NAME_KEY, "") or ""
    return PIISpan(start=r.start, end=r.end, text=text[r.start:r.end], entity_type=r.entity_type,
                   score=float(r.score), recognizer_name=name, metadata=meta)


class PIIAnonymizer:
    def __init__(self, engine: AnonymizerEngine | None = None, config=None, excluded_values=None,
                 included_entities=None, unmask_url: bool | None = None, span_variant_rules=None):
        from ..config.detector_config import AnonymizerConfig   # local: config imports masking
        cfg = config or AnonymizerConfig()
        self.engine = engine or AnonymizerEngine()
        self.config = cfg
        self.excluded_values = set(cfg.allow_listed_values if excluded_values is None else excluded_values)
        included = cfg.included_entities if included_entities is None else included_entities
        self.included_entities = set(included) if included is not None else None
        self.unmask_url = cfg.unmask_url if unmask_url is None else unmask_url
        self.span_variant_rules = dict(cfg.span_variant_rules if span_variant_rules is None else span_variant_rules)
        self.look_out_for = set(cfg.entities_to_look_out_for)
        self._separator_insensitive = frozenset(cfg.separator_insensitive)
        self.reset()

    # ---------- state ----------

    def reset(self) -> None:
        """Forget every identity and token issued so far, including loaded mappings."""
        self._counters: dict[str, int] = {}                          # entity -> last identity number
        self._identity: dict[tuple[str, str], int] = {}              # (entity, identity key) -> identity
        self._versions: dict[tuple[str, int], dict[str, int]] = {}   # (entity, identity) -> surface -> variant
        self._fixed: dict[str, str] = {}                             # user surface -> token
        self._observed: dict[str, str] = {}                          # surface -> token, first seen first
        self._selected: list[PIISpan] = []

    def rules(self, entity: str) -> tuple[str, ...]:
        return rules_for(entity, self._separator_insensitive, self.span_variant_rules)

    def load_mappings(self, mappings: dict[str, str]) -> None:
        """User-provided surface -> token pairs. These surfaces always get their given token, and tokens
        in the default shape ([PERSON_03], [PERSON_03_2]) reserve their numbers so new tokens never
        collide with them; other copies of the value share the identity."""
        for surface, token in mappings.items():
            self._fixed[surface] = token
            self._observed.setdefault(surface, token)
            try:
                entity, ident, variant = parse_token(token)
            except ValueError:
                continue
            if ident is None:
                continue
            self._identity.setdefault((entity, identity_key(surface, self.rules(entity))), ident)
            self._counters[entity] = max(self._counters.get(entity, 0), ident)
            if variant is not None:
                self._versions.setdefault((entity, ident), {}).setdefault(surface, variant)

    def mappings(self, text: str | None = None) -> dict[str, str]:
        """Surface -> token pairs generated or observed by this instance. With `text`, only surfaces that
        occur exactly (same Unicode string) in that text."""
        if text is None:
            return dict(self._observed)
        return {s: t for s, t in self._observed.items() if s and s in text}

    def token_mapping(self, text: str | None = None) -> dict[str, str]:
        """Token -> original surface. In original mode a token can stand for several surfaces; the
        first one seen is kept."""
        out: dict[str, str] = {}
        for surface, token in self.mappings(text).items():
            out.setdefault(token, surface)
        return out

    @property
    def selected_spans(self) -> list[PIISpan]:
        """The final maskable spans from the most recent resolve / anonymize call."""
        return list(self._selected)

    # ---------- resolve ----------

    def _keep(self, span: PIISpan, urls: list[tuple[int, int]]) -> bool:
        if span.text in self.excluded_values:
            return False
        if self.included_entities is not None and span.entity_type not in self.included_entities:
            return False
        if self.unmask_url and any(s <= span.start and span.end <= e for s, e in urls):
            return False
        return True

    def _identity_of(self, span: PIISpan) -> int:
        fixed = self._fixed.get(span.text)
        if fixed is not None:
            try:
                entity, ident, _ = parse_token(fixed)
                if ident is not None and entity == span.entity_type:
                    return ident
            except ValueError:
                pass
        key = (span.entity_type, identity_key(span.text, self.rules(span.entity_type)))
        if key not in self._identity:
            self._counters[span.entity_type] = self._counters.get(span.entity_type, 0) + 1
            self._identity[key] = self._counters[span.entity_type]
        return self._identity[key]

    @staticmethod
    def _free(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in occupied)

    def _propagate(self, text: str, spans: list[PIISpan], urls) -> list[PIISpan]:
        """Other copies of values in `entities_to_look_out_for`."""
        occupied = [(s.start, s.end) for s in spans]
        seen: set[tuple[str, str]] = set()
        added: list[PIISpan] = []
        sources = sorted((s for s in spans if s.entity_type in self.look_out_for),
                         key=lambda s: -(s.end - s.start))
        for src in sources:
            if (src.entity_type, src.text) in seen or len(src.text.strip()) < MIN_PROPAGATED_CHARS:
                continue
            seen.add((src.entity_type, src.text))
            for start, end in find_variants(text, src.text, self.rules(src.entity_type)):
                if not self._free(occupied, start, end) or is_inside_ascii_word(text, start, end):
                    continue
                span = PIISpan(start=start, end=end, text=text[start:end], entity_type=src.entity_type,
                               score=src.score, recognizer_name=f"propagated:{src.recognizer_name}",
                               metadata={"identity": src.metadata["identity"]})
                if not self._keep(span, urls):
                    continue
                occupied.append((start, end))
                added.append(span)
        return added

    def _extra_spans(self, text: str, spans: list[PIISpan]) -> list[PIISpan]:
        """Hook for subclasses: more spans, each with metadata['identity'] set. None by default."""
        return []

    @staticmethod
    def _merge_adjacent(text: str, spans: list[PIISpan]) -> list[PIISpan]:
        out: list[PIISpan] = []
        for span in spans:
            prev = out[-1] if out else None
            if (prev is not None and prev.entity_type == span.entity_type
                    and prev.metadata.get("identity") == span.metadata.get("identity")
                    and prev.end <= span.start and _HSPACE.fullmatch(text[prev.end:span.start])):
                out[-1] = replace(prev, end=span.end, text=text[prev.start:span.end],
                                  score=max(prev.score, span.score))
            else:
                out.append(span)
        return out

    def _token_for(self, span: PIISpan) -> str:
        if span.text in self._fixed:
            return self._fixed[span.text]
        ident = span.metadata["identity"]
        variant = 1
        if self.config.mode == "versioned":
            seen = self._versions.setdefault((span.entity_type, ident), {})
            if span.text not in seen:
                seen[span.text] = max(seen.values(), default=0) + 1
            variant = seen[span.text]
        return format_token(self.config.format, span.entity_type, ident, variant)

    def resolve(self, result: DetectionResult | None, text: str,
                native_results: list[RecognizerResult] | None = None) -> list[PIISpan]:
        """Assign identities and tokens and pick the final spans, without building masked text.
        Spans come from `native_results` (Presidio results) when given, else from `result.spans`."""
        t0 = time.perf_counter()
        if result is not None and result.checksum and result.checksum != checksum_of(text):
            raise ChecksumMismatchError("the detection result was computed on a different text")
        if native_results is not None:
            spans = [span_from_result(r, text) for r in native_results]
        else:
            spans = [replace(s, metadata=dict(s.metadata)) for s in (result.spans if result else [])]

        urls = [(m.start(), m.end()) for m in _URL.finditer(text)] if self.unmask_url else []
        spans = sorted((s for s in spans if self._keep(s, urls)), key=lambda s: (s.start, s.end))
        for span in spans:
            span.metadata["identity"] = self._identity_of(span)

        spans = spans + self._propagate(text, spans, urls)
        spans = spans + [s for s in self._extra_spans(text, spans) if self._keep(s, urls)]
        spans = self._merge_adjacent(text, sorted(spans, key=lambda s: (s.start, s.end)))

        for span in spans:
            span.token = self._token_for(span)
            self._observed.setdefault(span.text, span.token)
        self._selected = spans
        log.debug("resolve: %d spans in %.1f ms", len(spans), (time.perf_counter() - t0) * 1000)
        return list(spans)

    # ---------- anonymize ----------

    def _replacement(self, span: PIISpan) -> str:
        spec = self.config.per_entity.get(span.entity_type)
        if not spec:
            return span.token
        params = {k: v for k, v in spec.items() if k != "operator"}
        res = self.engine.anonymize(
            text=span.text,
            analyzer_results=[RecognizerResult(span.entity_type, 0, len(span.text), span.score)],
            operators={span.entity_type: OperatorConfig(spec["operator"], params)},
        )
        return res.text

    def anonymize(self, detection_result: DetectionResult | None, text: str,
                  native_results: list[RecognizerResult] | None = None) -> str:
        """Masked text: every selected span replaced by its token (or by its per-entity Presidio
        operator), including other copies of known values found by `resolve`."""
        spans = self.resolve(detection_result, text, native_results)
        out, pos = [], 0
        for span in spans:
            out.append(text[pos:span.start])
            out.append(self._replacement(span))
            pos = span.end
        out.append(text[pos:])
        return "".join(out)


def unmask(masked: str, mapping: dict[str, str]) -> str:
    """Restore original text from masked text and a token -> original mapping. Single pass, so text
    inserted by one replacement is never replaced again."""
    if not mapping:
        return masked
    pattern = re.compile("|".join(re.escape(p) for p in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group(0)], masked)
