"""Checks gate: user-defined rules are loaded from 'module:function', validated at load time, and gate
detections in the pipeline. The pipeline ships no rules; the rules here live in tests/check_rules.py."""
from datetime import date

import pytest

from pii_pipeline.checks import make_rule
from pii_pipeline.config import NerConfig, parse_config
from pii_pipeline.pipeline import PIIPipeline

TODAY = date(2025, 10, 30)


def rule(name, params=None):
    return make_rule(name, params or {})


# ---------- loading user rules ----------

def test_user_rule_is_called_with_text_params_and_today():
    seen = {}

    def spy(text, params, today):
        seen.update(text=text, params=params, today=today)
        return True
    import sys, types
    mod = types.ModuleType("spy_rules")
    mod.spy = spy
    sys.modules["spy_rules"] = mod
    try:
        assert rule("spy_rules:spy", {"k": 1})("abc", TODAY) is True
        assert seen == {"text": "abc", "params": {"k": 1}, "today": TODAY}
    finally:
        del sys.modules["spy_rules"]


def test_user_rule_return_value_is_coerced_to_bool():
    assert rule("check_rules:contains_word", {"word": "x"})("x", TODAY) is True
    assert rule("check_rules:contains_word", {"word": "x"})("y", TODAY) is False


def test_name_without_module_path_is_rejected():
    with pytest.raises(ValueError, match="package.module:function"):
        make_rule("min_age", {"years": 18})


def test_missing_module_or_function_fails_at_load_time():
    with pytest.raises(ValueError, match="invalid config"):
        parse_config({"checks": {"DOB": [{"rule": "no_such_module:fn"}]}})
    with pytest.raises(ValueError, match="invalid config"):
        parse_config({"checks": {"DOB": [{"rule": "check_rules:no_such_fn"}]}})


def test_valid_rule_passes_config_validation():
    cfg = parse_config({"checks": {"DOB": [{"rule": "check_rules:dob_adult", "years": 18}]}})
    assert cfg.checks[0].entity == "DOB" and cfg.checks[0].params == {"years": 18}


def test_check_entry_needs_a_rule_name():
    with pytest.raises(ValueError, match="rule name"):
        parse_config({"checks": {"DOB": [{"years": 18}]}})


def test_shipped_config_enables_no_checks():
    from pii_pipeline.config import load_config
    assert load_config("configs/pipeline.yaml").checks == []


# ---------- the gate in the pipeline ----------

def make_gated_pipeline(phrases, checks, deny=None):
    cfg = parse_config({"languages": ["en"], "checks": checks, "deny_list": deny or []})
    cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases})
    return PIIPipeline(cfg, today=TODAY)


ADULT = {"DOB": [{"rule": "check_rules:dob_adult", "years": 18}]}


def test_rejected_dob_is_not_masked_and_is_reported():
    p = make_gated_pipeline({"2020-05-01": "DOB", "1960-05-01": "DOB"}, ADULT)
    res = p.mask("Transfer on 2020-05-01. Born 1960-05-01.")
    assert res.text == "Transfer on 2020-05-01. Born [DOB_01]."
    assert [(r["entity_type"], r["rule"]) for r in res.rejected] == [("DOB", "check_rules:dob_adult")]


def test_rule_boundary_is_applied_by_the_user_rule():
    # the user's rule decides the boundary; the pipeline only applies the result
    p = make_gated_pipeline({"2007-10-30": "DOB", "2007-10-31": "DOB"}, ADULT)
    assert p.mask("2007-10-30 and 2007-10-31").text == "[DOB_01] and 2007-10-31"


def test_rejected_detection_carries_no_raw_text():
    p = make_gated_pipeline({"2020-05-01": "DOB"}, ADULT)
    rejected = p.mask("on 2020-05-01").rejected
    assert rejected and "text" not in rejected[0]


def test_rejected_span_does_not_block_a_merge_with_a_kept_span():
    p = make_gated_pipeline({"Tan 2020-05-01": "PERSON", "2020-05-01": "DOB"}, ADULT)
    assert p.mask("Tan 2020-05-01").text == "[PERSON_01]"


def test_deny_list_values_are_not_gated():
    p = make_gated_pipeline({}, ADULT, deny=[{"entity": "DOB", "values": ["2020-05-01"]}])
    assert p.mask("on 2020-05-01").text == "on [DOB_01]"


def test_labels_without_checks_are_unaffected():
    p = make_gated_pipeline({"2020-05-01": "PHONE"}, ADULT)
    assert p.mask("call 2020-05-01").text == "call [PHONE_01]"


def test_several_rules_on_one_label_all_must_pass():
    checks = {"DOB": [{"rule": "check_rules:not_future"},
                      {"rule": "check_rules:dob_adult", "years": 18}]}
    p = make_gated_pipeline({"2030-01-01": "DOB"}, checks)
    assert p.mask("on 2030-01-01").text == "on 2030-01-01"
    assert p.mask("on 2030-01-01").rejected[0]["rule"] == "check_rules:not_future"
