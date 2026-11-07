"""Span rules: checks a detected span must pass before it is masked.

A rule is a function fn(text, params, today) -> bool. True keeps (masks) the span, False rejects it.
`text` is the detected span, `params` the rule's settings from the config, `today` the reference date.

Config (`anonymizer.span_rules`), one list per label; "*" applies to every label:

    span_rules:
      "*": [{rule: min_length, chars: 2}]
      DOB: [{rule: dob_adult, years: 18}, {rule: "myrules.dates:not_future"}]

A rule is named by its short name (the functions in BUILTIN_RULES below) or by a
'package.module:function' path to your own function. A span is kept only if every rule for its label,
and every "*" rule, returns True. Rules fail open: a value they cannot parse is kept.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Callable

from ..registry import load_class

Rule = Callable[[str, date], bool]
ALL_LABELS = "*"

_ISO_DATE = re.compile(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*")
_DMY_DATE = re.compile(r"\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\s*")


def _parse_date(text: str) -> date | None:
    """ISO (YYYY-MM-DD) or day-first (DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY). None if neither parses."""
    try:
        m = _ISO_DATE.fullmatch(text)
        if m:
            return date(*map(int, m.groups()))
        m = _DMY_DATE.fullmatch(text)
        if m:
            d, mth, y = map(int, m.groups())
            return date(y, mth, d)
    except ValueError:
        return None
    return None


# ---------- built-in rules ----------

def min_length(text, params, today):
    """Keep spans with at least `chars` non-space characters (default 2)."""
    return len("".join(text.split())) >= int(params.get("chars", 2))


def max_length(text, params, today):
    """Keep spans with at most `chars` characters (default 200)."""
    return len(text) <= int(params.get("chars", 200))


def has_digit(text, params, today):
    """Keep spans with at least `count` digits (default 1). For account, phone and ID labels."""
    return sum(c.isdigit() for c in text) >= int(params.get("count", 1))


def has_letter(text, params, today):
    """Keep spans with at least one letter. Drops pure numbers and punctuation from name labels."""
    return any(c.isalpha() for c in text)


def matches(text, params, today):
    """Keep spans that fully match the regex `pattern`."""
    return re.fullmatch(params["pattern"], text.strip()) is not None


def not_in(text, params, today):
    """Reject spans equal (case-insensitively) to one of `values`."""
    value = " ".join(text.casefold().split())
    return value not in {" ".join(str(v).casefold().split()) for v in params.get("values", [])}


def dob_adult(text, params, today):
    """Keep dates at least `years` (default 18) before today. Unparseable dates are kept."""
    d = _parse_date(text)
    if d is None:
        return True
    age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return age >= int(params.get("years", 18))


def not_future(text, params, today):
    """Reject dates after today. Unparseable dates are kept."""
    d = _parse_date(text)
    return d is None or d <= today


BUILTIN_RULES: dict[str, Callable] = {
    "min_length": min_length,
    "max_length": max_length,
    "has_digit": has_digit,
    "has_letter": has_letter,
    "matches": matches,
    "not_in": not_in,
    "dob_adult": dob_adult,
    "not_future": not_future,
}


def make_rule(name: str, params: dict) -> Rule:
    """Resolve a rule by short name or 'package.module:function'. Unknown names, missing modules and
    missing functions fail here, so a bad config fails when it loads."""
    if ":" in name:
        fn = load_class(name)
    elif name in BUILTIN_RULES:
        fn = BUILTIN_RULES[name]
    else:
        raise ValueError(f"unknown rule {name!r}: use one of {sorted(BUILTIN_RULES)} "
                         f"or a 'package.module:function' path")
    if not callable(fn):
        raise ValueError(f"rule {name!r} is not callable")
    return lambda text, today: bool(fn(text, params, today))


class Gate:
    """Per-label rules. `checks` maps a label (or "*") to a list of (rule name, Rule)."""

    def __init__(self, checks: dict[str, list[tuple[str, Rule]]]):
        self.checks = checks

    def __len__(self) -> int:
        return sum(len(v) for v in self.checks.values())

    def rejected_by(self, label: str, text: str, today: date) -> str | None:
        """Name of the first failing rule, or None if the span passes every rule for its label."""
        for name, rule in [*self.checks.get(ALL_LABELS, []), *self.checks.get(label, [])]:
            if not rule(text, today):
                return name
        return None
