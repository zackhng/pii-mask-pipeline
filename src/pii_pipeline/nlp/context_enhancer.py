"""Checks on the words around a span.

is_inside_ascii_word   rejects a match that is a fragment of a longer ASCII word or identifier
                       ("Ann" inside "Annual", "nnual" inside "Annual report").
has_label_in_window    True if one of the given label words occurs within `window` characters of the span.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _ascii_word_char(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "_")


def is_inside_ascii_word(text: str, start: int, end: int) -> bool:
    """True if the span cuts an ASCII word: the character just before it and its first character are
    both ASCII word characters, or its last character and the character just after it are.
    Scripts written without spaces (Chinese, Japanese, Thai) are never treated as cut."""
    if start >= end:
        return False
    if start > 0 and _ascii_word_char(text[start - 1]) and _ascii_word_char(text[start]):
        return True
    if end < len(text) and _ascii_word_char(text[end]) and _ascii_word_char(text[end - 1]):
        return True
    return False


def _window(text: str, start: int, end: int, window: int) -> tuple[str, str]:
    return text[max(0, start - window):start], text[end:end + window]


def _contains_label(segment: str, label: str) -> bool:
    label = " ".join(label.casefold().split())
    if not label:
        return False
    # whole-word match for ASCII labels; substring for labels in unspaced scripts
    if label.isascii():
        return re.search(r"(?<![\w])" + re.escape(label).replace(r"\ ", r"\s+") + r"(?![\w])",
                         segment.casefold()) is not None
    return label in segment.casefold()


def has_label_in_window(text: str, start: int, end: int, labels, window: int = 40) -> bool:
    """True if any of `labels` occurs within `window` characters before or after the span."""
    before, after = _window(text, start, end, window)
    return any(_contains_label(before, lab) or _contains_label(after, lab) for lab in labels)


def has_any_label_in_surrounding(text: str, start: int, end: int, label_map: dict, window: int = 40) -> bool:
    """True if a label word of any entity in `label_map` ({ENTITY: [label words]}) is near the span."""
    return any(has_label_in_window(text, start, end, labels, window) for labels in label_map.values())


def neighbouring_words(text: str, start: int, end: int) -> tuple[str | None, str | None]:
    """The word directly before and directly after the span on the same line, separated from it only by
    spaces or tabs. None where there is no such word (start of line, punctuation, line break)."""
    before = None
    m = re.search(r"([^\W_]+)[ \t]+$", text[:start])
    if m and text[start - 1] in " \t":
        before = m.group(1)
    after = None
    m = re.match(r"[ \t]+([^\W_]+)", text[end:])
    if m:
        after = m.group(1)
    return before, after
