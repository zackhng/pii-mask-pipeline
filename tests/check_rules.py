"""User-style check rules used only by the tests. These are examples of what a template user writes;
the pipeline ships none. Each rule is fn(text, params, today) -> bool, True keeps the span."""
import re
from datetime import date


def dob_adult(text, params, today):
    """Keep only if the ISO date (YYYY-MM-DD) is at least `years` ago. Fails open on other formats."""
    m = re.fullmatch(r"\s*(\d{4})-(\d{2})-(\d{2})\s*", text)
    if not m:
        return True
    d = date(*map(int, m.groups()))
    age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return age >= params["years"]


def not_future(text, params, today):
    m = re.fullmatch(r"\s*(\d{4})-(\d{2})-(\d{2})\s*", text)
    return not m or date(*map(int, m.groups())) <= today


def contains_word(text, params, today):
    return params["word"] in text
