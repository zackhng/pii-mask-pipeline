"""Placeholder tokens that replace sensitive text.

Default formats:
    original   [PERSON_01]      one token per identity
    versioned  [PERSON_01_2]    one token per exact surface form: identity 01, variant 2

`parse_token` reads the default shapes back, plus the bare `[PERSON]` form. Custom formats can be
used for masking but are not parsed back.
"""
from __future__ import annotations

import re

MODES = ("original", "versioned")
DEFAULT_FORMATS = {
    "original": "[{entity}_{index:02d}]",
    "versioned": "[{entity}_{index:02d}_{version}]",
}

# entity: upper-case name, may contain underscores (EMAIL_ADDRESS); identity and variant: digits
_BODY = r"(?P<entity>[A-Z][A-Z0-9_]*?)(?:_(?P<identity>\d+)(?:_(?P<variant>\d+))?)?"
TOKEN_RE = re.compile(r"\[" + _BODY + r"\]")
_FULL_RE = re.compile(r"^\[" + _BODY + r"\]$")


def default_format(mode: str) -> str:
    return DEFAULT_FORMATS[mode]


def validate_format(fmt: str, mode: str) -> None:
    """Raise ValueError if `fmt` cannot build tokens for `mode`."""
    if mode not in MODES:
        raise ValueError(f"masking mode must be one of {list(MODES)}, got {mode!r}")
    if "{entity}" not in fmt or "{index" not in fmt:
        raise ValueError("format must contain {entity} and {index}")
    if mode == "versioned" and "{version" not in fmt:
        raise ValueError("format must contain {version} in versioned mode")
    try:
        fmt.format(entity="X", index=1, version=1)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"format {fmt!r} is not a valid template: {exc}") from exc


def format_token(fmt: str, entity: str, identity: int, variant: int = 1) -> str:
    return fmt.format(entity=entity, index=identity, version=variant)


def is_token(text: str) -> bool:
    return bool(_FULL_RE.match(text))


def parse_token(text: str) -> tuple[str, int | None, int | None]:
    """'[PERSON_01_2]' -> ('PERSON', 1, 2); '[PERSON_01]' -> ('PERSON', 1, None); '[PERSON]' -> ('PERSON', None, None).
    Raises ValueError for anything else."""
    m = _FULL_RE.match(text)
    if not m:
        raise ValueError(f"not a placeholder token: {text!r}")
    ident, var = m.group("identity"), m.group("variant")
    return m.group("entity"), (int(ident) if ident else None), (int(var) if var else None)


def find_tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, token) for every default-shaped token in `text`."""
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]
