"""Connect document labels to the values they introduce.

Tables (`table_semantics`):
    column  a header-row cell matching a column alias labels every non-empty cell below it
    row     the first cell of a row matching a row alias labels the other non-empty cells in that row
    Header cells and row-header cells are never labelled themselves.

Inline labels (`span_alias`), outside tables:
    colon   "Client name: Tan Wei Ming"
    point   "- DOB: 1990-01-01", "* Phone: +65 6123 4567", "1. Name: Tan"
    Several pairs may share a line ("Name: Tan Wei Ming; DOB: 1990-01-01"). A value runs to the end of
    the line, a ';' or '|', or the next known label, whichever comes first.

Aliases match exactly after normalising case, spacing and the * _ ` : emphasis around them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SEP_CELL = re.compile(r"^:?-+:?$")   # GFM: one or more dashes
_HEADER_STRIP = " \t*_`:："
_VALUE_STRIP_LEAD = " \t*_`"
_VALUE_STRIP_TAIL = " \t*_`,;.。，；"

# ---------- markdown tables ----------


@dataclass
class Cell:
    text: str     # trimmed cell content
    start: int    # char offsets in the original document
    end: int


@dataclass
class Table:
    header: list[Cell]
    rows: list[list[Cell]] = field(default_factory=list)
    start_line: int = 0   # line index of the header
    end_line: int = 0     # line index after the last row


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
                table = Table(header=header, start_line=i)
                j = i + 2
                while j < len(lines) and "|" in lines[j][1] and lines[j][1].strip():
                    table.rows.append(_split_cells(*lines[j]))
                    j += 1
                table.end_line = j
                tables.append(table)
                i = j
                continue
        i += 1
    return tables


# ---------- alias tables ----------


@dataclass
class TableFields:
    columns: dict[str, str]   # normalised column-header alias -> label
    rows: dict[str, str]      # normalised row-header alias -> label


def invert_aliases(section, name: str) -> dict[str, str]:
    """{LABEL: [alias, ...]} -> {normalised alias: LABEL}. An alias listed for two labels is an error."""
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
    return TableFields(columns=invert_aliases(column_headers, "table_semantics.columns"),
                       rows=invert_aliases(row_headers, "table_semantics.rows"))


# ---------- labelled values ----------


@dataclass
class LabelledValue:
    entity_type: str
    start: int
    end: int
    kind: str    # "table_columns" | "table_rows" | "inline_labels"


def table_values(text: str, fields: TableFields, mode: str) -> list[LabelledValue]:
    """Cells labelled by a column header (mode 'columns') or by a row header (mode 'rows')."""
    if mode not in ("columns", "rows"):
        raise ValueError(f"mode must be 'columns' or 'rows', got {mode!r}")
    kind = f"table_{mode}"
    out: list[LabelledValue] = []
    for table in find_tables(text):
        if mode == "columns":
            labels = [fields.columns.get(normalise_header(c.text)) for c in table.header]
            for row in table.rows:
                for j, label in enumerate(labels):
                    if label and j < len(row) and row[j].end > row[j].start:
                        out.append(LabelledValue(label, row[j].start, row[j].end, kind))
        else:
            for row in table.rows:
                label = fields.rows.get(normalise_header(row[0].text)) if row else None
                if label:
                    out.extend(LabelledValue(label, c.start, c.end, kind) for c in row[1:] if c.end > c.start)
    return out


# a label starts a line (after an optional bullet) or follows ';' / '|' / ','
_KEY_RE = re.compile(
    r"(?:^|(?<=[;|,，；]))[ \t]*(?:(?:[-*+•·]|\d{1,3}[.)])[ \t]+)?(?P<key>[^:：;|,，；\n]{1,60}?)[ \t]*[:：]")


def inline_values(text: str, aliases: dict[str, str]) -> list[LabelledValue]:
    """Values introduced by 'Label: value' outside tables. `aliases` maps normalised alias -> label."""
    if not aliases:
        return []
    table_lines: set[int] = set()
    for table in find_tables(text):
        table_lines.update(range(table.start_line, table.end_line))

    out: list[LabelledValue] = []
    for line_no, (line_start, line) in enumerate(_lines_with_offsets(text)):
        if line_no in table_lines or ":" not in line and "：" not in line:
            continue
        keys = [(m, aliases.get(normalise_header(m.group("key")))) for m in _KEY_RE.finditer(line)]
        known = [(m, label) for m, label in keys if label]
        for k, (m, label) in enumerate(known):
            value_end = known[k + 1][0].start() if k + 1 < len(known) else len(line)
            stop = re.search(r"[;|；]", line[m.end():value_end])
            if stop:
                value_end = m.end() + stop.start()
            s, e = m.end(), value_end
            while s < e and line[s] in _VALUE_STRIP_LEAD:
                s += 1
            while e > s and line[e - 1] in _VALUE_STRIP_TAIL:
                e -= 1
            if s < e and not line[s:e].startswith("//"):
                out.append(LabelledValue(label, line_start + s, line_start + e, "inline_labels"))
    return out
