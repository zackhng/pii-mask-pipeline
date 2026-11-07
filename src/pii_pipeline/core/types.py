"""Data passed between detection and masking."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def checksum_of(text: str) -> str:
    """sha256 of the text, used to check that a detection result is applied to the same text.
    'surrogatepass' keeps lone surrogates (valid in JSON "\\ud800" escapes) from raising."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass
class DetectionRequest:
    text: str
    context: dict[str, Any] = field(default_factory=dict)   # e.g. {"language": "zh"}; free-form
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = checksum_of(self.text)


@dataclass
class PIISpan:
    start: int
    end: int
    text: str                     # the detected surface; contains PII, never logged or returned by the API
    entity_type: str
    score: float
    recognizer_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token: str | None = None      # placeholder, set by the anonymizer

    @property
    def source(self) -> str:
        """'deny_list' | 'table_columns' | 'table_rows' | 'inline_labels' | 'patterns' | 'presidio' | 'ner' | ..."""
        return self.recognizer_name.split(":", 1)[0]


@dataclass
class DetectionResult:
    text: str
    checksum: str
    spans: list[PIISpan] = field(default_factory=list)       # final, non-overlapping, sorted by position
    metadata: dict[str, Any] = field(default_factory=dict)   # "rejected": spans the rules refused (no text)

    @property
    def rejected(self) -> list[dict]:
        return self.metadata.get("rejected", [])
