"""Markdown (GFM) table parsing and the table field definitions (column / row header aliases)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_SEP_CELL = re.compile(r"^:?-{3,}:?$")
_HEADER_STRIP = " \t*_`:"


@dataclass
class Cell:
    text: str     # trimmed cell content
    start: int    # char offsets in the original document
    end: int


@dataclass
class Table:
    header: list[Cell]
    rows: list[list[Cell]] = field(default_factory=list)


def normalise_header(text: str) -> str:
    """Match key for headers and aliases: strip emphasis / trailing colon, casefold, collapse spaces."""
    return " ".join(text.strip(_HEADER_STRIP).casefold().split())


def _lines_with_offsets(text: str):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line.rstrip("\r\n")
        pos += len(line)


def _split_cells(line_start: int, line: str) -> list[Cell]:
    # an unescaped '|' separates cells; '\|' stays inside its cell
    bounds = [i for i, c in enumerate(line) if c == "|" and (i == 0 or line[i - 1] != "\\")]
    edges = [-1, *bounds, len(line)]
    segments = [(edges[k] + 1, edges[k + 1]) for k in range(len(edges) - 1)]
    # leading / trailing pipes produce empty outer segments; drop them
    if line.lstrip().startswith("|") and segments and not line[segments[0][0]:segments[0][1]].strip():
        segments = segments[1:]
    if line.rstrip().endswith("|") and segments and not line[segments[-1][0]:segments[-1][1]].strip():
        segments = segments[:-1]
    cells = []
    for s, e in segments:
        while s < e and line[s].isspace():
            s += 1
        while e > s and line[e - 1].isspace():
            e -= 1
        cells.append(Cell(text=line[s:e], start=line_start + s, end=line_start + e))
    return cells


def _is_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_cells(0, line)
    return bool(cells) and all(_SEP_CELL.match(c.text) for c in cells)


def find_tables(text: str) -> list[Table]:
    """All GFM tables in `text`: a header line with '|', a separator line, then consecutive '|' lines."""
    lines = list(_lines_with_offsets(text))
    tables: list[Table] = []
    i = 0
    while i < len(lines) - 1:
        start, head = lines[i]
        _, sep = lines[i + 1]
        if "|" in head and _is_separator(sep):
            header = _split_cells(start, head)
            if len(_split_cells(lines[i + 1][0], sep)) == len(header):
                table = Table(header=header)
                j = i + 2
                while j < len(lines) and "|" in lines[j][1] and lines[j][1].strip():
                    table.rows.append(_split_cells(*lines[j]))
                    j += 1
                tables.append(table)
                i = j
                continue
        i += 1
    return tables


@dataclass
class TableFields:
    columns: dict[str, str]   # normalised column-header alias -> label
    rows: dict[str, str]      # normalised row-header alias -> label


def _invert(section, name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, aliases in (section or {}).items():
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases or []:
            key = normalise_header(str(alias))
            if not key:
                raise ValueError(f"{name}: empty alias for {label}")
            if out.get(key, label) != label:
                raise ValueError(f"{name}: alias {alias!r} is listed for both {out[key]} and {label}")
            out[key] = label
    return out


def table_fields_from_sections(column_headers, row_headers) -> TableFields:
    return TableFields(columns=_invert(column_headers, "column_headers"),
                       rows=_invert(row_headers, "row_headers"))


def load_table_fields(path: str | Path) -> TableFields:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return table_fields_from_sections(raw.get("column_headers"), raw.get("row_headers"))
