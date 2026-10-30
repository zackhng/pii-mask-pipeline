"""Rule-based checks: a gate that a detected span must pass before it is masked.

Rules are defined by the person using the template, never shipped here. A rule is a Python function
in the user's own module, referenced from the config as 'package.module:function':

    def dob_adult(text, params, today):   # text: the detected span; today: date
        ...
        return True                        # keep (mask) the span; False rejects it

A span is kept only if every rule for its label returns True. Rules receive the span text and the
reference date, so they can check formats and values however the domain needs.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from .registry import load_class

Rule = Callable[[str, date], bool]


def make_rule(name: str, params: dict) -> Rule:
    """Resolve a user-defined rule from 'package.module:function'. Fails if the module or function
    does not exist, so a bad config is caught at load time."""
    if ":" not in name:
        raise ValueError(f"rule {name!r} must be a 'package.module:function' path to a user-defined function")
    fn = load_class(name)
    if not callable(fn):
        raise ValueError(f"rule {name!r} is not callable")
    return lambda text, today: bool(fn(text, params, today))


class Gate:
    """Per-label checks. `checks` maps a label to a list of (rule name, Rule)."""

    def __init__(self, checks: dict[str, list[tuple[str, Rule]]]):
        self.checks = checks

    def rejected_by(self, label: str, text: str, today: date) -> str | None:
        """Name of the first failing rule, or None if the span passes every check for its label."""
        for name, rule in self.checks.get(label, []):
            if not rule(text, today):
                return name
        return None
