"""Regression tests for bugs found by stress-testing hostile inputs."""
import time

import pytest
from fastapi.testclient import TestClient

from pii_pipeline import unmask
from pii_pipeline.api import create_app
from pii_pipeline.config import NerConfig, parse_config
from pii_pipeline.core.errors import ConfigError
from pii_pipeline.core.intervals import IntervalSet
from pii_pipeline.core.types import DetectionResult, PIISpan, checksum_of
from pii_pipeline.masking.optimized_anonymizer import OptimizedAnonymizer, name_variants
from pii_pipeline.pipeline import PIIPipeline


def pipe(phrases=None, **raw):
    cfg = parse_config({"languages": ["en"], **raw})
    if phrases is not None:
        cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases})
    return PIIPipeline(cfg)


# ---------- leaks ----------

def test_model_boundary_one_letter_short_masks_the_whole_word():
    res = pipe({"Tan Wei Min": "PERSON"}).mask("Hello Tan Wei Ming.")
    assert res.text == "Hello [PERSON_01]." and res.rejected == []


def test_value_glued_to_a_label_is_masked():
    assert pipe({"12345678": "ACCOUNT"}).mask("acct12345678").text == "[ACCOUNT_01]"


def test_small_fragment_of_a_word_is_still_rejected():
    res = pipe({"Ann": "PERSON"}).mask("Annual report")
    assert res.text == "Annual report" and res.rejected[0]["rule"] == "context:inside_word"


def test_separated_value_is_not_matched_as_prefix_of_a_longer_one():
    p = pipe({"123-456": "ACCOUNT"}, anonymizer={"entities_to_look_out_for": ["ACCOUNT"],
                                                 "separator_insensitive": ["ACCOUNT"]})
    out = p.mask("Account 123-456 is ours. Account 123-456-7 is not. Ref 0-123-456.").text
    assert out == "Account [ACCOUNT_01] is ours. Account [ACCOUNT_02] is not. Ref [ACCOUNT_03]."


def test_separated_value_keeps_its_sentence_full_stop():
    p = pipe({"123-456": "ACCOUNT"}, anonymizer={"separator_insensitive": ["ACCOUNT"]})
    assert p.mask("Paid from 123-456. Thanks").text == "Paid from [ACCOUNT_01]. Thanks"


def test_included_entities_apply_before_cross_type_merge():
    p = pipe({"Tan Wei Ming": "PERSON", "Tan Wei Ming Holdings": "BUSINESS"},
             anonymizer={"included_entities": ["PERSON"]})
    assert p.mask("Tan Wei Ming Holdings").text == "[PERSON_01] Holdings"


def test_tables_with_one_dash_delimiters():
    p = pipe(None, anonymizer={"table_semantics": {"columns": [{"name": "account no", "entity_type": "ACCOUNT"}]}})
    assert p.mask("| Account No |\n|-|\n| 123456 |").text == "| Account No |\n|-|\n| [ACCOUNT_01] |"


def test_long_inline_values_are_masked():
    addr = "Block 123 " + "Very Long Street Name " * 10 + "Singapore 123456"
    p = pipe(None, anonymizer={"span_alias": {"ADDRESS": ["address"]}})
    assert p.mask(f"Address: {addr}").text == "Address: [ADDRESS_01]"


# ---------- crashes ----------

def test_lone_surrogate_does_not_crash_the_pipeline():
    assert pipe({"Tan Wei Ming": "PERSON"}).mask("Tan Wei Ming \ud800").text == "[PERSON_01] \ud800"


def test_lone_surrogate_is_a_422_in_the_api(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("languages: [en]\n", encoding="utf-8")
    with TestClient(create_app(config_path=str(cfg))) as c:
        r = c.post("/mask", content=b'{"text": "a \\ud800"}', headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["code"] == "invalid_text"


@pytest.mark.parametrize("raw", [
    {"anonymizer": "x"},
    {"presidio_recognizers": []},
    {"presidio_recognizers": {"deny_list": ["Bob"]}},
    {"anonymizer": {"span_rules": [{"rule": "min_length"}]}},
    {"anonymizer": {"span_alias": ["name"]}},
    {"anonymizer": {"table_semantics": [{"name": "x"}]}},
    {"model": {"gliner": ["a"]}},
    {"regex_patterns": [{"label": "X", "name": "x", "pattern": "("}]},
    {"anonymizer": {"per_entity": {"DOB": {"operator": "redcat"}}}},
    {"anonymizer": {"per_entity": {"DOB": {"operator": "encrypt"}}}},           # missing key
    {"anonymizer": {"span_rules": {"X": [{"rule": "matches"}]}}},              # missing pattern
    {"anonymizer": {"span_rules": {"X": [{"rule": "min_length", "chars": "two"}]}}},
    {"ner": {"class": "fake_ner:FakeNER", "threshold": "high"}},
    {"regex_patterns": [{"label": "X", "name": "x", "pattern": "a", "context": ["b"], "context_window": "far"}]},
    {"ner": {"model": "m", "window_size": "big"}, "model": {"token_classification": {"m": {"model_dir": "x"}}}},
])
def test_bad_config_types_and_values_fail_at_load(raw):
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_user_rule_that_raises_keeps_the_span(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("boom_rules")
    mod.boom = lambda text, params, today: 1 / 0
    monkeypatch.setitem(sys.modules, "boom_rules", mod)
    p = pipe({"Bob": "PERSON"}, anonymizer={"span_rules": {"PERSON": [{"rule": "boom_rules:boom"}]}})
    assert p.mask("Bob").text == "[PERSON_01]"


# ---------- over-masking ----------

def test_single_word_name_does_not_mask_the_lowercase_word():
    p = pipe({"Will": "PERSON"}, anonymizer={"entities_to_look_out_for": ["PERSON"]})
    assert p.mask("Will Tan called. Will said he will call.").text == \
        "[PERSON_01] Tan called. [PERSON_01] said he will call."


def _opt(text, names):
    spans = sorted((PIISpan(i, i + len(n), n, "PERSON", 0.9, "ner:F") for n in names
                    for i in [text.find(n)] if i != -1), key=lambda s: s.start)
    return OptimizedAnonymizer().anonymize(DetectionResult(text, checksum_of(text), spans), text)


def test_lowercase_name_particles_are_not_variants():
    assert "bin" not in name_variants("Ahmad bin Ali") and "de" not in name_variants("Van de Berg")
    out = _opt("Ahmad bin Ali signed. Put it in the bin. Van de Berg met de Jong.", ["Ahmad bin Ali", "Van de Berg"])
    assert "in the bin" in out and "met de Jong" in out


def test_surname_in_a_firm_name_is_not_a_variant():
    assert _opt("Tan Wei Ming works at Tan & Partners.", ["Tan Wei Ming"]) == "[PERSON_01] works at Tan & Partners."


# ---------- performance ----------

def test_interval_set():
    s = IntervalSet([(0, 3), (10, 20)])
    assert s.is_free(3, 10) and not s.is_free(2, 4) and not s.is_free(15, 16) and not s.is_free(5, 25)
    s.add(5, 8)
    assert not s.is_free(7, 9) and s.is_free(8, 10)


def test_max_size_document_is_fast():
    names = {"Tan Wei Ming": "PERSON", "123-456789-001": "ACCOUNT", "Acme Pte Ltd": "BUSINESS"}
    text = ("Tan Wei Ming paid Acme Pte Ltd from 123-456789-001. " * 4000)[:200_000]
    p = pipe(names, anonymizer={"entities_to_look_out_for": ["PERSON", "ACCOUNT", "BUSINESS"],
                                "separator_insensitive": ["ACCOUNT"], "optimized": True, "mode": "versioned"})
    t0 = time.perf_counter()
    res = p.mask(text, return_mapping=True)
    assert time.perf_counter() - t0 < 5
    assert unmask(res.text, res.mapping) == text
