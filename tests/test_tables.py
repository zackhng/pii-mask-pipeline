"""Markdown table parsing and table-field labelling (column headers and row headers)."""
import pytest

from pii_pipeline.config import NerConfig, TableConfig, parse_config
from pii_pipeline.pipeline import PIIPipeline
from pii_pipeline.tables import find_tables, load_table_fields, normalise_header


def write_fields(tmp_path, body: str):
    path = tmp_path / "fields.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


FIELDS = """
column_headers:
  ACCOUNT: [account no, acct]
  PHONE: [phone]
row_headers:
  PERSON: [client name]
  DOB: [date of birth]
"""


def make_table_pipeline(tmp_path, fields_body=FIELDS, phrases=None):
    cfg = parse_config({"languages": ["en"]})
    cfg.tables = TableConfig(fields_file=write_fields(tmp_path, fields_body))
    if phrases:
        cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases})
    return PIIPipeline(cfg)


def test_normalise_header_ignores_emphasis_colon_and_spacing():
    assert normalise_header("**Account  No.**:") == "account  no.".replace("  ", " ")
    assert normalise_header(" _Phone_ ") == "phone"


def test_cell_offsets_point_at_cell_text():
    text = "intro\n\n| Account No | Name |\n|---|---|\n|  123-456  | Tan |\n\nafter"
    (table,) = find_tables(text)
    assert [c.text for c in table.header] == ["Account No", "Name"]
    assert [c.text for c in table.rows[0]] == ["123-456", "Tan"]
    for cell in table.header + table.rows[0]:
        assert text[cell.start:cell.end] == cell.text


def test_escaped_pipe_does_not_split_cells():
    (table,) = find_tables("| a \\| b | c |\n|---|---|\n| 1 | 2 |")
    assert [c.text for c in table.header] == ["a \\| b", "c"]


def test_rows_without_separator_are_not_a_table():
    assert find_tables("| a | b |\n| 1 | 2 |") == []


def test_table_ends_at_blank_line():
    (table,) = find_tables("| a | b |\n|---|---|\n| 1 | 2 |\n\n| 3 | 4 |")
    assert len(table.rows) == 1


def test_duplicate_alias_for_two_labels_is_an_error(tmp_path):
    path = write_fields(tmp_path, "column_headers:\n  ACCOUNT: [acct]\n  PHONE: [acct]\n")
    with pytest.raises(ValueError, match="acct"):
        load_table_fields(path)


def test_alias_as_plain_string_is_accepted(tmp_path):
    fields = load_table_fields(write_fields(tmp_path, "column_headers:\n  PHONE: phone\n"))
    assert fields.columns == {"phone": "PHONE"}


def test_column_header_labels_cells_below_and_not_the_header(tmp_path):
    p = make_table_pipeline(tmp_path)
    text = "| Account No | Note |\n|---|---|\n| 123-456789-001 | Account No |\n| 555 | x |"
    out = p.mask(text).text
    assert out == "| Account No | Note |\n|---|---|\n| [ACCOUNT_01] | Account No |\n| [ACCOUNT_02] | x |"


def test_row_header_labels_other_cells_in_the_row(tmp_path):
    p = make_table_pipeline(tmp_path)
    text = "| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |\n| Date of birth | 16/09/1923 |"
    out = p.mask(text).text
    assert out == ("| Field | Value |\n|---|---|\n| Client name | [PERSON_01] |\n"
                   "| Date of birth | [DOB_01] |")


def test_row_header_cell_itself_is_not_masked(tmp_path):
    p = make_table_pipeline(tmp_path)
    out = p.mask("| Client name | Tan |\n|---|---|\n| Client name | x |").text
    assert out.splitlines()[2].startswith("| Client name |")


def test_table_labels_take_priority_over_ner_on_the_same_span(tmp_path):
    # NER calls "Tan Wei Ming" a BUSINESS; the table says it is a PERSON. The table label wins.
    p = make_table_pipeline(tmp_path, phrases={"Tan Wei Ming": "BUSINESS"})
    out = p.mask("| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |").text
    assert out == "| Field | Value |\n|---|---|\n| Client name | [PERSON_01] |"


def test_table_label_extends_over_ner_span_without_leaking_remainder(tmp_path):
    # NER finds only "Tan"; the table covers the whole cell, so the merged span masks all of it.
    p = make_table_pipeline(tmp_path, phrases={"Tan": "PERSON"})
    out = p.mask("| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |").text
    assert "Wei Ming" not in out and "Tan" not in out


def test_table_fields_are_listed_in_entities_with_source(tmp_path):
    p = make_table_pipeline(tmp_path)
    ent = p.mask("| Account No |\n|---|\n| 123 |").entities[0]
    assert ent["source"] == "table_columns" and ent["entity_type"] == "ACCOUNT"


def test_empty_cells_are_skipped(tmp_path):
    p = make_table_pipeline(tmp_path)
    out = p.mask("| Account No | Phone |\n|---|---|\n|  | +65 6123 4567 |").text
    # the empty account cell is left alone; the phone cell is labelled through the "Phone" column header
    assert out == "| Account No | Phone |\n|---|---|\n|  | [PHONE_01] |"


def test_config_requires_fields_file_or_inline_headers():
    with pytest.raises(ValueError, match="fields_file, or column_headers"):
        parse_config({"tables": {"score": 0.9}})


def test_config_rejects_both_file_and_inline_headers(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        parse_config({"tables": {"fields_file": "x.yaml", "column_headers": {"PHONE": ["phone"]}}})


def test_inline_table_headers_work_without_a_file():
    cfg = parse_config({"languages": ["en"], "tables": {"column_headers": {"ACCOUNT": ["account no"]}}})
    p = PIIPipeline(cfg)
    assert p.mask("| Account No |\n|---|\n| 123 |").text == "| Account No |\n|---|\n| [ACCOUNT_01] |"
