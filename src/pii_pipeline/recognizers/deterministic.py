"""Deterministic recognizers built from YAML: exact-value deny lists and regex patterns.
Both are stock Presidio PatternRecognizers."""
from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer

from ..config import DenyListItem, PatternItem

_CASE_INSENSITIVE = re.DOTALL | re.MULTILINE | re.IGNORECASE
_CASE_SENSITIVE = re.DOTALL | re.MULTILINE


def deny_list_recognizers(items: list[DenyListItem]) -> list[PatternRecognizer]:
    return [
        PatternRecognizer(
            supported_entity=item.entity,
            name=f"deny_list:{item.entity}:{i}",
            deny_list=item.values,
            deny_list_score=1.0,
            global_regex_flags=_CASE_SENSITIVE if item.case_sensitive else _CASE_INSENSITIVE,
        )
        for i, item in enumerate(items)
    ]


def pattern_recognizers(items: list[PatternItem]) -> list[PatternRecognizer]:
    return [
        PatternRecognizer(
            supported_entity=item.entity,
            name=f"patterns:{item.name}",
            patterns=[Pattern(name=item.name, regex=item.regex, score=item.score)],
        )
        for item in items
    ]
