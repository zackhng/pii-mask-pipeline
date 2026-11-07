"""tokens.py, span_variants.py and context_enhancer.py."""
import pytest

from pii_pipeline.masking.span_variants import find_variants, identity_key, rules_for
from pii_pipeline.masking.tokens import find_tokens, format_token, is_token, parse_token, validate_format
from pii_pipeline.nlp.context_enhancer import (
    has_any_label_in_surrounding,
    has_label_in_window,
    is_inside_ascii_word,
    neighbouring_words,
)

# ---------- tokens ----------


def test_format_and_parse_round_trip():
    tok = format_token("[{entity}_{index:02d}_{version}]", "PERSON", 3, 2)
    assert tok == "[PERSON_03_2]"
    assert parse_token(tok) == ("PERSON", 3, 2)
    assert parse_token("[PERSON_01]") == ("PERSON", 1, None)
    assert parse_token("[PERSON]") == ("PERSON", None, None)


def test_entity_names_with_underscores_parse():
    assert parse_token("[EMAIL_ADDRESS_02_1]") == ("EMAIL_ADDRESS", 2, 1)
    assert parse_token("[EMAIL_ADDRESS_02]") == ("EMAIL_ADDRESS", 2, None)


def test_non_tokens_are_rejected():
    for bad in ("PERSON_01", "[person_01]", "[PERSON_01.2]", "[]"):
        assert not is_token(bad)
    with pytest.raises(ValueError):
        parse_token("[PERSON_01.2]")


def test_find_tokens_in_text():
    assert [t for _, _, t in find_tokens("Hi [PERSON_01], ref [ACCOUNT_02_1].")] == ["[PERSON_01]", "[ACCOUNT_02_1]"]


def test_validate_format():
    validate_format("[{entity}_{index:02d}]", "original")
    with pytest.raises(ValueError, match="version"):
        validate_format("[{entity}_{index:02d}]", "versioned")
    with pytest.raises(ValueError, match="entity"):
        validate_format("[PII]", "original")
    with pytest.raises(ValueError, match="valid template"):
        validate_format("[{entity}_{index:02d}_{nope}]", "original")


# ---------- span variants ----------


def test_default_rules_ignore_case_and_whitespace():
    rules = rules_for("PERSON")
    assert identity_key("Tan  Wei\nMing", rules) == identity_key("TAN WEI MING", rules)


def test_separator_rule_only_for_configured_labels():
    assert identity_key("123-456 789", rules_for("ACCOUNT", ["ACCOUNT"])) == "123456789"
    assert identity_key("123-456 789", rules_for("ADDRESS", ["ACCOUNT"])) == "123-456 789"


def test_email_split_across_lines_is_one_identity():
    rules = rules_for("EMAIL")
    assert identity_key("john.doe@exam\nple.com", rules) == identity_key("John.Doe@example.com", rules)


def test_find_variants_finds_split_and_reformatted_copies():
    text = "Mail john.doe@example.com or john.doe@exam\n  ple.com today"
    found = [text[s:e] for s, e in find_variants(text, "john.doe@example.com", rules_for("EMAIL"))]
    assert found == ["john.doe@example.com", "john.doe@exam\n  ple.com"]

    text = "acct 123-456-789 and 123456789 and 123 456 789"
    found = [text[s:e] for s, e in find_variants(text, "123456789", rules_for("ACCOUNT", ["ACCOUNT"]))]
    assert found == ["123-456-789", "123456789", "123 456 789"]


def test_explicit_rule_override():
    assert rules_for("PERSON", overrides={"PERSON": ["whitespace"]}) == ("whitespace",)
    assert identity_key("Tan", ("whitespace",)) != identity_key("TAN", ("whitespace",))


# ---------- context enhancer ----------


def test_inside_ascii_word():
    text = "Annual report by Ann"
    assert is_inside_ascii_word(text, 0, 3)            # "Ann" in "Annual"
    assert is_inside_ascii_word(text, 1, 6)            # "nnual"
    assert not is_inside_ascii_word(text, 17, 20)      # the whole word "Ann"
    assert not is_inside_ascii_word("陈伟明先生", 0, 3)   # unspaced scripts are never cut


def test_label_in_window():
    text = "Tel: +65 6123 4567, ref 99"
    assert has_label_in_window(text, 5, 18, ["tel", "phone"])
    assert not has_label_in_window(text, 24, 26, ["phone"], window=5)
    assert not has_label_in_window("Hotel 123", 6, 9, ["tel"])   # whole words only
    assert has_any_label_in_surrounding(text, 5, 18, {"PHONE": ["tel"], "DOB": ["dob"]})


def test_neighbouring_words():
    text = "Mr Tan Ah Kow, then Tan."
    assert neighbouring_words(text, 3, 6) == ("Mr", "Ah")
    assert neighbouring_words(text, 20, 23) == ("then", None)
