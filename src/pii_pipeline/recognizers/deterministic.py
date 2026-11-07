"""Deterministic recognizers built from config: exact-value deny lists, regex patterns, and Presidio's
predefined recognizers."""
from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer import predefined_recognizers

from ..config.detector_config import DenyListItem, PatternItem
from ..core.errors import ConfigError
from ..nlp.context_enhancer import has_label_in_window

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


class ContextPatternRecognizer(PatternRecognizer):
    """A regex recognizer whose matches need one of `context_words` within `context_window` characters.
    Works without an NLP engine, unlike Presidio's token-based context enhancement."""

    def __init__(self, *args, context_words: list[str], context_window: int = 40, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_words = list(context_words)
        self.context_window = context_window

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)
        return [r for r in results
                if has_label_in_window(text, r.start, r.end, self.context_words, self.context_window)]


def pattern_recognizers(items: list[PatternItem]) -> list[PatternRecognizer]:
    out = []
    for item in items:
        kwargs = dict(supported_entity=item.entity, name=f"patterns:{item.name}",
                      patterns=[Pattern(name=item.name, regex=item.regex, score=item.score)])
        if item.context:
            out.append(ContextPatternRecognizer(**kwargs, context_words=item.context,
                                                context_window=item.context_window))
        else:
            out.append(PatternRecognizer(**kwargs))
    return out


def presidio_builtin_recognizers(names: list[str]) -> list:
    """Presidio's predefined recognizers by class name, e.g. EmailRecognizer."""
    out = []
    for name in names:
        cls = getattr(predefined_recognizers, name, None)
        if cls is None or not isinstance(cls, type):
            raise ConfigError(f"presidio_recognizers.builtin: {name!r} is not a Presidio predefined recognizer")
        try:
            rec = cls()
        except Exception as exc:
            raise ConfigError(f"presidio_recognizers.builtin: {name} could not be created: {exc}") from exc
        rec.name = f"presidio:{name}"
        out.append(rec)
    return out
