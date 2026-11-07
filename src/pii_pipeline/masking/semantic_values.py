"""Let document structure override model detection.

A table cell or inline-label value says what its content is ("Client name | Tan Wei Ming"). Where a
model span (NER, regex pattern, Presidio built-in) overlaps such a structured span, the structured
span wins on the text it covers:

- a model span inside a structured span is dropped; the structured label is used
- a model span that reaches outside it keeps only the part outside (if that part has any letters or
  digits), so a model span never joins two table cells into one placeholder

Deny-list spans are explicit user assertions and are left to the normal overlap merge.
"""
from __future__ import annotations

from presidio_analyzer import RecognizerResult

STRUCTURED_SOURCES = frozenset({"table_columns", "table_rows", "inline_labels"})
MODEL_SOURCES = frozenset({"ner", "patterns", "presidio"})
_TRIM = " \t\r\n|"


def _source(r: RecognizerResult) -> str:
    name = (r.recognition_metadata or {}).get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
    return name.split(":", 1)[0]


def _piece(text: str, r: RecognizerResult, start: int, end: int) -> RecognizerResult | None:
    while start < end and text[start] in _TRIM:
        start += 1
    while end > start and text[end - 1] in _TRIM:
        end -= 1
    if start >= end or not any(c.isalnum() for c in text[start:end]):
        return None
    return RecognizerResult(r.entity_type, start, end, r.score, recognition_metadata=r.recognition_metadata)


def apply_semantic_overrides(text: str, results: list[RecognizerResult]) -> list[RecognizerResult]:
    structured = sorted(((r.start, r.end) for r in results if _source(r) in STRUCTURED_SOURCES))
    if not structured:
        return list(results)

    out: list[RecognizerResult] = []
    for r in results:
        if _source(r) not in MODEL_SOURCES:
            out.append(r)
            continue
        pieces = [(r.start, r.end)]
        for s, e in structured:
            if e <= r.start or s >= r.end:
                continue
            next_pieces = []
            for a, b in pieces:
                if e <= a or s >= b:
                    next_pieces.append((a, b))
                    continue
                if a < s:
                    next_pieces.append((a, s))
                if e < b:
                    next_pieces.append((e, b))
            pieces = next_pieces
        if pieces == [(r.start, r.end)]:
            out.append(r)
        else:
            out.extend(p for p in (_piece(text, r, a, b) for a, b in pieces) if p is not None)
    return out
