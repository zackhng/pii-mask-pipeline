"""Table-field recognizer: labels the cell values of markdown tables by their column or row headers.

mode "columns": a header-row cell matching a column alias labels every non-empty cell below it.
mode "rows":    the first cell of a row matching a row alias labels the other non-empty cells in that row.
Header cells and row-header cells are never labelled themselves.
"""
from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

from ..tables import Cell, TableFields, find_tables, normalise_header


class TableFieldRecognizer(EntityRecognizer):
    def __init__(self, fields: TableFields, mode: str, score: float = 0.9, supported_language: str = "en"):
        if mode not in ("columns", "rows"):
            raise ValueError(f"mode must be 'columns' or 'rows', got {mode!r}")
        self.mode = mode
        self.score = score
        self.fields = fields
        aliases = self.fields.columns if mode == "columns" else self.fields.rows
        super().__init__(supported_entities=sorted(set(aliases.values())),
                         supported_language=supported_language, name=f"table_{mode}")

    def load(self) -> None:
        self.is_loaded = True

    def analyze(self, text, entities, nlp_artifacts=None):
        out: list[RecognizerResult] = []
        for table in find_tables(text):
            if self.mode == "columns":
                labels = [self.fields.columns.get(normalise_header(c.text)) for c in table.header]
                for row in table.rows:
                    for j, label in enumerate(labels):
                        if label and j < len(row):
                            out.append(self._emit(label, row[j], entities))
            else:
                for row in table.rows:
                    if not row:
                        continue
                    label = self.fields.rows.get(normalise_header(row[0].text))
                    if label:
                        for cell in row[1:]:
                            out.append(self._emit(label, cell, entities))
        return [r for r in out if r is not None]

    def _emit(self, label: str, cell: Cell, entities):
        if cell.end <= cell.start or (entities and label not in entities):
            return None
        return RecognizerResult(
            entity_type=label, start=cell.start, end=cell.end, score=self.score,
            recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name},
        )
