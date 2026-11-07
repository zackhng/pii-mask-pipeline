"""Structured labels: markdown tables (table_semantics) and inline 'Label: value' lines (span_alias)."""
import pytest

from pii_pipeline.config import NerConfig, parse_config
from pii_pipeline.masking.structured_rules import find_tables, inline_values, invert_aliases, normalise_header
from pii_pipeline.pipeline import PIIPipeline

TABLES = {
    "columns": [{"name": "account no", "entity_type": "ACCOUNT", "aliases": ["acct"]},
                {"name": "phone", "entity_type": "PHONE"}],
    "rows": [{"name": "client name", "entity_type": "PERSON"},
             {"name": "date of birth", "entity_type": "DOB"}],
}


def make_pipeline(tables=TABLES, span_alias=None, phrases=None):
    anonymizer = {}
    if tables:
        anonymizer["table_semantics"] = tables
    if span_alias:
        anonymizer["span_alias"] = span_alias
    cfg = parse_config({"languages": ["en"], "anonymizer": anonymizer})
    if phrases:
        cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases})
    return PIIPipeline(cfg)


# ---------- parsing ----------

def test_normalise_header_ignores_emphasis_colon_and_spacing():
    assert normalise_header("**Account  No.**:") == "account no."
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


def test_duplicate_alias_for_two_labels_is_an_error():
    with pytest.raises(ValueError, match="acct"):
        invert_aliases({"ACCOUNT": ["acct"], "PHONE": ["acct"]}, "x")
    with pytest.raises(ValueError, match="acct"):
        parse_config({"anonymizer": {"table_semantics": {"columns": [
            {"name": "acct", "entity_type": "ACCOUNT"}, {"name": "x", "entity_type": "PHONE", "aliases": ["acct"]}]}}})


def test_alias_as_plain_string_is_accepted():
    assert invert_aliases({"PHONE": "phone"}, "x") == {"phone": "PHONE"}
    cfg = parse_config({"anonymizer": {"span_alias": {"PHONE": "tel"}}})
    assert cfg.anonymizer.inline_aliases() == {"tel": "PHONE"}


def test_table_semantics_needs_columns_or_rows_and_entity_type():
    with pytest.raises(ValueError, match="columns or rows"):
        parse_config({"anonymizer": {"table_semantics": {"score": 0.9}}})
    with pytest.raises(ValueError, match="entity_type"):
        parse_config({"anonymizer": {"table_semantics": {"columns": [{"name": "phone"}]}}})


# ---------- tables in the pipeline ----------

def test_column_header_labels_cells_below_and_not_the_header():
    text = "| Account No | Note |\n|---|---|\n| 123-456789-001 | Account No |\n| 555 | x |"
    out = make_pipeline().mask(text).text
    assert out == "| Account No | Note |\n|---|---|\n| [ACCOUNT_01] | Account No |\n| [ACCOUNT_02] | x |"


def test_field_name_and_aliases_both_match():
    out = make_pipeline().mask("| Acct |\n|---|\n| 123 |").text
    assert out == "| Acct |\n|---|\n| [ACCOUNT_01] |"


def test_row_header_labels_other_cells_in_the_row():
    text = "| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |\n| Date of birth | 16/09/1923 |"
    out = make_pipeline().mask(text).text
    assert out == ("| Field | Value |\n|---|---|\n| Client name | [PERSON_01] |\n"
                   "| Date of birth | [DOB_01] |")


def test_row_header_cell_itself_is_not_masked():
    out = make_pipeline().mask("| Client name | Tan |\n|---|---|\n| Client name | x |").text
    assert out.splitlines()[2].startswith("| Client name |")


def test_table_labels_take_priority_over_ner_on_the_same_span():
    # NER calls "Tan Wei Ming" a BUSINESS; the table says it is a PERSON. The table label wins.
    p = make_pipeline(phrases={"Tan Wei Ming": "BUSINESS"})
    out = p.mask("| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |").text
    assert out == "| Field | Value |\n|---|---|\n| Client name | [PERSON_01] |"


def test_table_label_extends_over_ner_span_without_leaking_remainder():
    # NER finds only "Tan"; the table covers the whole cell, so the cell is masked whole.
    p = make_pipeline(phrases={"Tan": "PERSON"})
    out = p.mask("| Field | Value |\n|---|---|\n| Client name | Tan Wei Ming |").text
    assert "Wei Ming" not in out and "Tan" not in out


def test_model_span_crossing_cells_is_cut_at_the_labelled_cell():
    # NER joins two cells; the labelled cell keeps its table label, the rest keeps the NER label.
    p = make_pipeline(phrases={"123-456 | Acme Holdings": "BUSINESS"})
    out = p.mask("| Account No | Employer |\n|---|---|\n| 123-456 | Acme Holdings |").text
    assert out.splitlines()[2] == "| [ACCOUNT_01] | [BUSINESS_01] |"


def test_table_fields_are_listed_in_entities_with_source():
    ent = make_pipeline().mask("| Account No |\n|---|\n| 123 |").entities[0]
    assert ent["source"] == "table_columns" and ent["entity_type"] == "ACCOUNT"


def test_empty_cells_are_skipped():
    out = make_pipeline().mask("| Account No | Phone |\n|---|---|\n|  | +65 6123 4567 |").text
    assert out == "| Account No | Phone |\n|---|---|\n|  | [PHONE_01] |"


# ---------- inline labels ----------

ALIASES = {"client name": "PERSON", "name": "PERSON", "dob": "DOB", "phone": "PHONE"}


def values(text):
    return [(v.entity_type, text[v.start:v.end]) for v in inline_values(text, ALIASES)]


def test_colon_and_bullet_forms():
    text = "Client name: Tan Wei Ming\n- DOB: 1990-01-01\n* **Phone:** +65 6123 4567\n1. Name： 陈伟明"
    assert values(text) == [("PERSON", "Tan Wei Ming"), ("DOB", "1990-01-01"),
                            ("PHONE", "+65 6123 4567"), ("PERSON", "陈伟明")]


def test_several_pairs_on_one_line():
    assert values("Name: Tan Wei Ming; DOB: 1990-01-01, Phone: 6123 4567.") == [
        ("PERSON", "Tan Wei Ming"), ("DOB", "1990-01-01"), ("PHONE", "6123 4567")]


def test_unknown_labels_urls_and_tables_are_ignored():
    assert values("Note: call back\nSee https://example.com/name: x") == []
    assert values("| Name: Tan |\n|---|\n| x |") == []


def test_inline_labels_mask_in_the_pipeline_and_override_ner():
    p = make_pipeline(tables=None, span_alias={"PERSON": ["client name"]}, phrases={"Tan Wei": "BUSINESS"})
    res = p.mask("Client name: Tan Wei Ming\nOther text")
    assert res.text == "Client name: [PERSON_01]\nOther text"
    assert res.entities[0]["source"] == "inline_labels"
