"""Unit tests with a fake NER: deny lists, patterns, allow list, overlap priority, indexed masking."""
import pytest
from presidio_analyzer import RecognizerResult

from pii_pipeline.analyzer import merge_overlaps
from pii_pipeline.config import AnonymizerConfig, NerConfig, parse_config
from pii_pipeline.core.types import DetectionResult, PIISpan, checksum_of
from pii_pipeline.masking.anonymizer import PIIAnonymizer
from pii_pipeline.masking.span_variants import normalise
from pii_pipeline.pipeline import PIIPipeline


def make_pipeline(phrases=None, deny=None, patterns=None, allow=None, priority=None, masking=None,
                  separator_insensitive=None):
    anonymizer = {"allow_listed_values": allow or [], "separator_insensitive": separator_insensitive or [],
                  **(masking or {})}
    if priority:
        anonymizer["priority"] = priority
    raw = {
        "languages": ["en"],
        "presidio_recognizers": {"deny_list": deny or []},
        "regex_patterns": patterns or [],
        "anonymizer": anonymizer,
    }
    cfg = parse_config(raw)
    if phrases is not None:
        cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases})
    return PIIPipeline(cfg)


def test_placeholders_are_indexed_and_consistent():
    p = make_pipeline(phrases={"Tan Wei Ming": "PERSON", "Lee Kuan": "PERSON"})
    out = p.mask("Tan Wei Ming met Lee Kuan. Then Tan Wei Ming left.").text
    assert out == "[PERSON_01] met [PERSON_02]. Then [PERSON_01] left."


def test_indices_follow_position_not_entity_order():
    p = make_pipeline(phrases={"Bob": "PERSON", "Acme Pte Ltd": "BUSINESS"})
    out = p.mask("Acme Pte Ltd hired Bob, then Bob thanked Acme Pte Ltd.").text
    assert out == "[BUSINESS_01] hired [PERSON_01], then [PERSON_01] thanked [BUSINESS_01]."


def test_case_insensitive_deny_list_matches_and_is_indexed_as_same_value():
    p = make_pipeline(deny=[{"entity": "BUSINESS", "values": ["Acme Wealth Pte Ltd"]}])
    out = p.mask("acme wealth pte ltd and ACME WEALTH PTE LTD").text
    assert out == "[BUSINESS_01] and [BUSINESS_01]"


def test_case_sensitive_deny_list_respects_case():
    p = make_pipeline(deny=[{"entity": "PERSON", "values": ["Jane Doe"], "case_sensitive": True}])
    assert p.mask("Jane Doe").text == "[PERSON_01]"
    assert p.mask("jane doe").text == "jane doe"


def test_regex_pattern():
    p = make_pipeline(patterns=[{"name": "nric", "label": "TIN", "pattern": r"\b[STFG]\d{7}[A-Z]\b", "score": 0.85}])
    assert p.mask("NRIC S1234567D on file").text == "NRIC [TIN_01] on file"


def test_allow_list_is_not_masked():
    p = make_pipeline(phrases={"Our Firm Pte Ltd": "BUSINESS", "Acme": "BUSINESS"}, allow=["Our Firm Pte Ltd"])
    assert p.mask("Our Firm Pte Ltd and Acme").text == "Our Firm Pte Ltd and [BUSINESS_01]"


def test_separator_insensitive_account_numbers_share_an_index():
    p = make_pipeline(phrases={"123-456789-001": "ACCOUNT", "123456789001": "ACCOUNT"},
                      separator_insensitive=["ACCOUNT"])
    out = p.mask("a 123-456789-001 b 123456789001").text
    assert out == "a [ACCOUNT_01] b [ACCOUNT_01]"


def test_mapping_only_when_requested():
    p = make_pipeline(phrases={"Bob": "PERSON"})
    assert p.mask("Bob").mapping is None
    assert p.mask("Bob", return_mapping=True).mapping == {"[PERSON_01]": "Bob"}


def test_entities_have_no_raw_text():
    p = make_pipeline(phrases={"Bob": "PERSON"})
    ent = p.mask("Hi Bob").entities[0]
    assert ent["placeholder"] == "[PERSON_01]" and ent["source"] == "ner" and "text" not in ent


def test_per_entity_operator_override():
    p = make_pipeline(phrases={"Bob": "PERSON", "Acme": "BUSINESS"},
                      masking={"per_entity": {"BUSINESS": {"operator": "redact"}}})
    assert p.mask("Bob and Acme").text == "[PERSON_01] and "


def test_masking_format_is_configurable():
    p = make_pipeline(phrases={"Bob": "PERSON"}, masking={"format": "<{entity}#{index}>"})
    assert p.mask("Bob").text == "<PERSON#1>"


def _res(entity, start, end, score, name):
    return RecognizerResult(entity, start, end, score, recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: name})


PRIORITY = ["deny_list", "patterns", "ner"]


def test_overlap_merges_to_union_with_priority_label():
    deny = _res("BUSINESS", 0, 4, 1.0, "deny_list:BUSINESS:0")
    ner = _res("ORG_LONG", 0, 10, 0.99, "ner:X")
    merged = merge_overlaps([ner, deny], PRIORITY)
    assert [(r.entity_type, r.start, r.end) for r in merged] == [("BUSINESS", 0, 10)]


def test_overlap_label_by_score_within_same_source():
    a = _res("PERSON", 0, 5, 0.6, "ner:X")
    b = _res("ORGANIZATION", 3, 8, 0.9, "ner:X")
    merged = merge_overlaps([a, b], PRIORITY)
    assert [(r.entity_type, r.start, r.end) for r in merged] == [("ORGANIZATION", 0, 8)]


def test_overlap_is_transitive():
    spans = [_res("PERSON", 0, 4, 0.9, "ner:X"), _res("PERSON", 3, 7, 0.9, "ner:X"),
             _res("PERSON", 6, 10, 0.9, "ner:X")]
    assert [(r.start, r.end) for r in merge_overlaps(spans, PRIORITY)] == [(0, 10)]


def test_touching_spans_are_not_merged():
    a = _res("PERSON", 0, 3, 0.9, "ner:X")
    b = _res("PERSON", 3, 6, 0.9, "ner:X")
    assert [(r.start, r.end) for r in merge_overlaps([b, a], PRIORITY)] == [(0, 3), (3, 6)]


def test_deny_list_inside_ner_span_no_longer_leaks_remainder():
    p = make_pipeline(phrases={"Acme Wealth Pte Ltd": "BUSINESS"}, deny=[{"entity": "BUSINESS", "values": ["Acme"]}])
    assert p.mask("Acme Wealth Pte Ltd is here").text == "[BUSINESS_01] is here"


def test_variants_share_a_placeholder_and_normalise():
    text = "bob, BOB, Bob"
    spans = [PIISpan(s, s + 3, text[s:s + 3], "PERSON", 1.0, "ner:X") for s in (0, 5, 10)]
    result = DetectionResult(text, checksum_of(text), spans)
    resolved = PIIAnonymizer(config=AnonymizerConfig()).resolve(result, text)
    assert [s.token for s in resolved] == ["[PERSON_01]"] * 3
    assert normalise("12-3 4", "ACCOUNT", frozenset({"ACCOUNT"})) == "1234"
    assert normalise("12-3 4", "ADDRESS", frozenset({"ACCOUNT"})) == "12-3 4"
    assert normalise("12-3 4", "ACCOUNT") == "12-3 4"   # not separator-insensitive unless configured


def test_config_rejects_bad_priority_and_format():
    with pytest.raises(ValueError, match="priority"):
        parse_config({"anonymizer": {"priority": ["ner", "spacy"]}})
    with pytest.raises(ValueError, match="priority"):
        parse_config({"anonymizer": {"priority": ["ner", "ner"]}})
    with pytest.raises(ValueError, match="anonymizer.format"):
        parse_config({"anonymizer": {"format": "[PII]"}})
    with pytest.raises(ValueError, match="score"):
        parse_config({"regex_patterns": [{"name": "x", "label": "TIN", "pattern": "a", "score": 1.5}]})


def test_partial_priority_ranks_unlisted_sources_last():
    cfg = parse_config({"anonymizer": {"priority": ["ner", "deny_list"]}})
    assert cfg.anonymizer.priority[:2] == ["ner", "deny_list"] and len(cfg.anonymizer.priority) == 7


def test_old_top_level_keys_point_to_their_new_place():
    with pytest.raises(ValueError, match="anonymizer.span_rules"):
        parse_config({"checks": {}})
    with pytest.raises(ValueError, match="presidio_recognizers.deny_list"):
        parse_config({"deny_list": []})
