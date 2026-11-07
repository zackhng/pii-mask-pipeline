"""Structured-label recognizers: Presidio wrappers around masking/structured_rules.py.

StructuredLabelRecognizer(mode="columns" | "rows")  table cells labelled by column or row headers
StructuredLabelRecognizer(mode="inline")            values after "Label:" outside tables
"""
from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

from ..masking.structured_rules import TableFields, inline_values, table_values

NAMES = {"columns": "table_columns", "rows": "table_rows", "inline": "inline_labels"}


class StructuredLabelRecognizer(EntityRecognizer):
    def __init__(self, mode: str, score: float = 0.9, fields: TableFields | None = None,
                 aliases: dict[str, str] | None = None, supported_language: str = "en"):
        if mode not in NAMES:
            raise ValueError(f"mode must be one of {list(NAMES)}, got {mode!r}")
        self.mode = mode
        self.score = score
        self.fields = fields or TableFields(columns={}, rows={})
        self.aliases = dict(aliases or {})
        labels = {"columns": self.fields.columns, "rows": self.fields.rows, "inline": self.aliases}[mode]
        super().__init__(supported_entities=sorted(set(labels.values())),
                         supported_language=supported_language, name=NAMES[mode])

    def load(self) -> None:
        self.is_loaded = True

    def analyze(self, text, entities, nlp_artifacts=None):
        if self.mode == "inline":
            found = inline_values(text, self.aliases)
        else:
            found = table_values(text, self.fields, self.mode)
        return [
            RecognizerResult(entity_type=v.entity_type, start=v.start, end=v.end, score=self.score,
                             recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name})
            for v in found if not entities or v.entity_type in entities
        ]


def TableFieldRecognizer(fields: TableFields, mode: str, score: float = 0.9, supported_language: str = "en"):
    """Kept for callers of the old class: a structured recognizer for table columns or rows."""
    return StructuredLabelRecognizer(mode, score=score, fields=fields, supported_language=supported_language)
