"""PIIPipeline: detect -> span rules -> context checks -> structure overrides -> merge -> anonymize.

detect():
    1. Presidio analyzer runs every detector (deny lists, tables, inline labels, regex, Presidio
       built-ins, NER). The allow list is applied here.
    2. Span rules: each span must pass the rules for its label and the "*" rules. Deny-list values
       are not checked.
    3. Context: a model or regex span that is a fragment of a longer ASCII word is rejected.
    4. Table and inline labels override model spans on the text they cover (semantic_values.py).
    5. Overlapping spans are merged into their union.
mask():
    6. A fresh (Optimized)PIIAnonymizer assigns identities and tokens and builds the masked text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from presidio_anonymizer import AnonymizerEngine

from .analyzer import build_analyzer, merge_overlaps, source_of
from .config.detector_config import DetectorConfig, load_config
from .core.errors import UnsupportedLanguageError
from .core.types import DetectionRequest, DetectionResult, PIISpan
from .masking.anonymizer import PIIAnonymizer, span_from_result
from .masking.optimized_anonymizer import OptimizedAnonymizer
from .masking.semantic_values import apply_semantic_overrides
from .masking.span_rules import Gate, make_rule
from .nlp.context_enhancer import is_inside_ascii_word

UNGATED_SOURCES = frozenset({"deny_list"})
WORD_CHECKED_SOURCES = frozenset({"ner", "patterns", "presidio"})


@dataclass
class MaskResult:
    text: str
    entities: list[dict] = field(default_factory=list)   # type, span, score, source, placeholder (no raw text)
    mapping: dict[str, str] | None = None                 # placeholder -> original; only if requested
    rejected: list[dict] = field(default_factory=list)   # detections refused by span rules (no raw text)


def rejected_dict(r, rule: str) -> dict:
    source = r.source if isinstance(r, PIISpan) else source_of(r)
    return {"entity_type": r.entity_type, "start": r.start, "end": r.end, "source": source, "rule": rule}


def entity_dict(span: PIISpan, placeholder: str | None = None) -> dict:
    out = {"entity_type": span.entity_type, "start": span.start, "end": span.end,
           "score": round(float(span.score), 4), "source": span.source}
    if placeholder is not None:
        out["placeholder"] = placeholder
    return out


class PIIPipeline:
    def __init__(self, config: DetectorConfig, today: date | None = None):
        self.config = config
        self.analyzer = build_analyzer(config)
        self.engine = AnonymizerEngine()
        self.today = today   # fixed reference date for date rules; None = the real date at each call
        self.gate = Gate(self._build_rules(config))

    @staticmethod
    def _build_rules(config: DetectorConfig) -> dict:
        rules: dict[str, list] = {}
        for item in config.anonymizer.span_rules:
            rules.setdefault(item.entity, []).append((item.rule, make_rule(item.rule, item.params)))
        return rules

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, today: date | None = None) -> "PIIPipeline":
        return cls(load_config(path), today=today)

    def new_anonymizer(self) -> PIIAnonymizer:
        cls = OptimizedAnonymizer if self.config.anonymizer.optimized else PIIAnonymizer
        return cls(self.engine, self.config.anonymizer)

    def detect(self, request: str | DetectionRequest, language: str | None = None) -> DetectionResult:
        """Detection only. Nothing is masked."""
        if isinstance(request, str):
            request = DetectionRequest(text=request)
        text = request.text
        language = language or request.context.get("language") or self.config.default_language
        if language not in self.config.languages:
            raise UnsupportedLanguageError(
                f"language {language!r} is not configured; available: {self.config.languages}")

        raw = self.analyzer.analyze(text=text, language=language,
                                    allow_list=self.config.anonymizer.allow_listed_values,
                                    allow_list_match="exact")

        # rules run before merging, so a rejected span never masks text
        today = self.today or date.today()
        kept, rejected = [], []
        for r in raw:
            source = source_of(r)
            rule = None
            if source not in UNGATED_SOURCES:
                rule = self.gate.rejected_by(r.entity_type, text[r.start:r.end], today)
            if rule is None and source in WORD_CHECKED_SOURCES and is_inside_ascii_word(text, r.start, r.end):
                rule = "context:inside_word"
            if rule:
                rejected.append(rejected_dict(r, rule))
            else:
                kept.append(r)

        kept = apply_semantic_overrides(text, kept)
        merged = merge_overlaps(kept, self.config.anonymizer.priority)
        return DetectionResult(text=text, checksum=request.checksum,
                               spans=[span_from_result(r, text) for r in merged],
                               metadata={"rejected": rejected, "language": language})

    def mask(self, text: str | DetectionRequest, language: str | None = None, return_mapping: bool = False,
             anonymizer: PIIAnonymizer | None = None) -> MaskResult:
        """Detect and mask. Pass a long-lived `anonymizer` to keep tokens consistent across documents."""
        found = self.detect(text, language)
        anon = anonymizer or self.new_anonymizer()
        masked = anon.anonymize(found, found.text)
        spans = anon.selected_spans
        mapping = anon.token_mapping(found.text) if return_mapping else None
        if mapping is not None:
            used = {s.token for s in spans}
            mapping = {tok: surf for tok, surf in mapping.items() if tok in used}
        return MaskResult(text=masked, entities=[entity_dict(s, s.token) for s in spans],
                          mapping=mapping, rejected=found.rejected)
