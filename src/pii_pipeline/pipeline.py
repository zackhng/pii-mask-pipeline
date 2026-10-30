"""PIIPipeline: Presidio analyzer (NER + deterministic recognizers) -> checks gate -> merge -> masking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from .analyzer import build_analyzer, merge_overlaps, source_of
from .checks import Gate, make_rule
from .config import PipelineConfig, load_config
from .masking import mask_text


@dataclass
class MaskResult:
    text: str
    entities: list[dict] = field(default_factory=list)   # type, span, score, source, placeholder (no raw text)
    mapping: dict[str, str] | None = None                 # placeholder -> original; only if requested
    rejected: list[dict] = field(default_factory=list)   # detections the checks gate refused (no raw text)


@dataclass
class DetectResult:
    entities: list[RecognizerResult]   # after the gate and merging, sorted by position
    rejected: list[dict]               # refused by the checks gate (no raw text)


def rejected_dict(r: RecognizerResult, rule: str) -> dict:
    return {"entity_type": r.entity_type, "start": r.start, "end": r.end, "source": source_of(r), "rule": rule}


def entity_dict(r: RecognizerResult, placeholder: str | None = None) -> dict:
    out = {"entity_type": r.entity_type, "start": r.start, "end": r.end,
           "score": round(float(r.score), 4), "source": source_of(r)}
    if placeholder is not None:
        out["placeholder"] = placeholder
    return out


class PIIPipeline:
    def __init__(self, config: PipelineConfig, today: date | None = None):
        self.config = config
        self.analyzer = build_analyzer(config)
        self.anonymizer = AnonymizerEngine()
        self.today = today   # fixed reference date for date checks; None = the real date at each call
        self.gate = Gate(self._build_checks(config))

    @staticmethod
    def _build_checks(config: PipelineConfig) -> dict[str, list[tuple[str, object]]]:
        checks: dict[str, list[tuple[str, object]]] = {}
        for item in config.checks:
            checks.setdefault(item.entity, []).append((item.rule, make_rule(item.rule, item.params)))
        return checks

    @classmethod
    def from_yaml(cls, path: str | Path, today: date | None = None) -> "PIIPipeline":
        return cls(load_config(path), today=today)

    def detect(self, text: str, language: str | None = None) -> DetectResult:
        """Detection only: NER + deterministic recognizers, gate, merge. Nothing is masked."""
        language = language or self.config.default_language
        raw = self.analyzer.analyze(
            text=text,
            language=language,
            allow_list=self.config.allow_list,
            allow_list_match="exact",
        )

        # Gate: each detection must pass the checks for its label. Deny-list values are explicit user
        # assertions and are not gated. Gating runs before merging, so a rejected span never masks text.
        today = self.today or date.today()
        kept, rejected = [], []
        for r in raw:
            rule = None if source_of(r) == "deny_list" else self.gate.rejected_by(r.entity_type, text[r.start:r.end], today)
            if rule:
                rejected.append(rejected_dict(r, rule))
            else:
                kept.append(r)

        return DetectResult(entities=merge_overlaps(kept, self.config.priority), rejected=rejected)

    def mask(self, text: str, language: str | None = None, return_mapping: bool = False) -> MaskResult:
        found = self.detect(text, language)
        masked, mapping, placeholders = mask_text(
            text, found.entities, self.config.masking, self.anonymizer, return_mapping)
        entities = [entity_dict(r, ph) for r, ph in zip(found.entities, placeholders)]
        return MaskResult(text=masked, entities=entities, mapping=mapping, rejected=found.rejected)
