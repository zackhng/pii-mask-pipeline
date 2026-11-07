"""Recognise differently formatted versions of one sensitive value as one identity.

Each label has a list of variant rules:
    case        "Tan Wei Ming" == "TAN WEI MING"
    whitespace  runs of spaces / tabs / line breaks are equal ("Tan  Wei\\nMing" == "Tan Wei Ming")
    separators  spaces, dots and hyphens are ignored ("123-456 789" == "123456789")
    linebreaks  a value may be split across lines with no space ("john.doe@exam\\nple.com"), as e-mail
                addresses often are when text is wrapped

Two things use the rules:
    identity_key()     decides whether two detected surfaces are the same value (same placeholder index)
    variant_pattern()  a regex that finds other formatted copies of a known value in the text
"""
from __future__ import annotations

import re

RULES = ("case", "whitespace", "separators", "linebreaks")
DEFAULT_RULES = ("case", "whitespace")
EMAIL_LABELS = frozenset({"EMAIL", "EMAIL_ADDRESS"})

_SEPARATORS = re.compile(r"[\s.\-]")
_BREAK = r"(?:[ \t]*\r?\n[ \t]*)?"


def rules_for(entity: str, separator_insensitive=(), overrides: dict | None = None) -> tuple[str, ...]:
    """The variant rules for a label: an explicit override, else case + whitespace, plus separators for
    `separator_insensitive` labels and line breaks for e-mail labels."""
    if overrides and entity in overrides:
        return tuple(overrides[entity])
    rules = list(DEFAULT_RULES)
    if entity in separator_insensitive:
        rules.append("separators")
    if entity in EMAIL_LABELS:
        rules.append("linebreaks")
    return tuple(rules)


def identity_key(text: str, rules) -> str:
    value = text
    if "case" in rules:
        value = value.casefold()
    if "linebreaks" in rules:
        value = re.sub(r"[ \t]*\r?\n[ \t]*", "", value)
    if "whitespace" in rules:
        value = " ".join(value.split())
    if "separators" in rules:
        value = _SEPARATORS.sub("", value)
    return value


def normalise(text: str, entity: str, separator_insensitive=frozenset()) -> str:
    """Identity key with the default rules. Kept for callers of the old masking.normalise."""
    return identity_key(text, rules_for(entity, separator_insensitive))


def variant_pattern(surface: str, rules) -> re.Pattern | None:
    """Regex matching `surface` in any format the rules treat as equal. None for an empty surface."""
    key = identity_key(surface, [r for r in rules if r != "case"])
    if not key:
        return None
    parts: list[str] = []
    for i, ch in enumerate(key):
        if ch.isspace():
            parts.append(r"\s+" if "whitespace" in rules else re.escape(ch))
            continue
        if i and parts and not key[i - 1].isspace():
            if "separators" in rules:
                parts.append(r"[\s.\-]?")
            elif "linebreaks" in rules:
                parts.append(_BREAK)
        parts.append(re.escape(ch))
    flags = re.IGNORECASE if "case" in rules else 0
    return re.compile("".join(parts), flags)


def find_variants(text: str, surface: str, rules) -> list[tuple[int, int]]:
    """(start, end) of every formatted copy of `surface` in `text`."""
    pattern = variant_pattern(surface, rules)
    if pattern is None:
        return []
    return [(m.start(), m.end()) for m in pattern.finditer(text) if m.end() > m.start()]
