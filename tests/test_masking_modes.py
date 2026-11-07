"""Masking modes: original (one placeholder per value) and versioned (one placeholder per exact text,
mapping 1:1 back to the original)."""
import pytest
from fastapi.testclient import TestClient

from pii_pipeline import unmask
from pii_pipeline.api import create_app
from pii_pipeline.config import NerConfig, parse_config
from pii_pipeline.pipeline import PIIPipeline


def make(mode=None, phrases=None, separator_insensitive=None, fmt=None):
    anonymizer = {"separator_insensitive": separator_insensitive or []}
    if mode:
        anonymizer["mode"] = mode
    if fmt:
        anonymizer["format"] = fmt
    cfg = parse_config({"languages": ["en"], "anonymizer": anonymizer})
    cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": phrases or {}})
    return PIIPipeline(cfg)


TEXT = "Tan Wei Ming met tan wei ming, then Tan Wei Ming (陈伟明) called TAN WEI MING."
PHRASES = {"Tan Wei Ming": "PERSON", "tan wei ming": "PERSON", "TAN WEI MING": "PERSON",
           "陈伟明": "PERSON"}


# ---------- original mode (unchanged behaviour) ----------

def test_original_mode_shares_one_placeholder_across_variants():
    res = make("original", PHRASES).mask(TEXT, return_mapping=True)
    # 陈伟明 is a different value from the English name, so it gets its own index
    assert res.text == "[PERSON_01] met [PERSON_01], then [PERSON_01] ([PERSON_02]) called [PERSON_01]."
    assert set(res.mapping) == {"[PERSON_01]", "[PERSON_02]"}


def test_original_mode_mapping_keeps_only_the_first_variant():
    res = make("original", PHRASES).mask("tan wei ming and Tan Wei Ming", return_mapping=True)
    assert res.mapping == {"[PERSON_01]": "tan wei ming"}   # lossy: "Tan Wei Ming" is not recoverable


def test_original_is_the_default_mode():
    cfg = parse_config({})
    assert cfg.anonymizer.mode == "original" and cfg.anonymizer.format == "[{entity}_{index:02d}]"


# ---------- versioned mode ----------

def test_versioned_mode_gives_each_exact_text_its_own_version():
    res = make("versioned", PHRASES).mask(TEXT, return_mapping=True)
    assert res.text == ("[PERSON_01_1] met [PERSON_01_2], then [PERSON_01_1] ([PERSON_02_1]) called [PERSON_01_3].")


def test_versioned_mapping_is_one_to_one():
    res = make("versioned", PHRASES).mask(TEXT, return_mapping=True)
    assert res.mapping == {
        "[PERSON_01_1]": "Tan Wei Ming",
        "[PERSON_01_2]": "tan wei ming",
        "[PERSON_01_3]": "TAN WEI MING",
        "[PERSON_02_1]": "陈伟明",
    }
    assert len(set(res.mapping.values())) == len(res.mapping)


def test_versioned_round_trip_restores_the_original_text_exactly():
    pipeline = make("versioned", PHRASES)
    res = pipeline.mask(TEXT, return_mapping=True)
    assert unmask(res.text, res.mapping) == TEXT


def test_versioned_separator_variants_share_an_index_with_separate_versions():
    phrases = {"123-456789-001": "ACCOUNT", "123456789001": "ACCOUNT"}
    res = make("versioned", phrases, separator_insensitive=["ACCOUNT"]).mask(
        "a 123-456789-001 b 123456789001", return_mapping=True)
    assert res.text == "a [ACCOUNT_01_1] b [ACCOUNT_01_2]"
    assert res.mapping == {"[ACCOUNT_01_1]": "123-456789-001", "[ACCOUNT_01_2]": "123456789001"}


def test_versioned_default_format_has_the_version():
    cfg = parse_config({"anonymizer": {"mode": "versioned"}})
    assert cfg.anonymizer.format == "[{entity}_{index:02d}_{version}]"


def test_versioned_with_custom_format_keeps_version():
    res = make("versioned", {"Bob": "PERSON"}, fmt="<{entity}#{index}.{version}>").mask("Bob", return_mapping=True)
    assert res.text == "<PERSON#1.1>"


def test_entities_report_the_versioned_placeholder():
    res = make("versioned", PHRASES).mask("Tan Wei Ming and tan wei ming")
    assert [e["placeholder"] for e in res.entities] == ["[PERSON_01_1]", "[PERSON_01_2]"]


def test_mapping_is_absent_unless_requested_in_both_modes():
    assert make("versioned", PHRASES).mask("Tan Wei Ming").mapping is None
    assert make("original", PHRASES).mask("Tan Wei Ming").mapping is None


# ---------- unmask helper ----------

def test_unmask_is_a_single_pass():
    # an original value that itself looks like a placeholder must not be replaced again
    mapping = {"[PERSON_01]": "[PERSON_02]", "[PERSON_02]": "Bob"}
    assert unmask("[PERSON_01]", mapping) == "[PERSON_02]"


def test_unmask_with_no_mapping_returns_input():
    assert unmask("plain text", {}) == "plain text"


# ---------- config validation ----------

def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="anonymizer.mode"):
        parse_config({"anonymizer": {"mode": "reversible"}})


def test_versioned_format_without_version_is_rejected():
    with pytest.raises(ValueError, match="{version}"):
        parse_config({"anonymizer": {"mode": "versioned", "format": "[{entity}_{index:02d}]"}})


# ---------- API ----------

CONFIG = """
languages: [en]
anonymizer:
  mode: versioned
ner:
  class: fake_ner:FakeNER
  params:
    phrases:
      "Tan Wei Ming": PERSON
      "tan wei ming": PERSON
"""


def test_api_reports_masking_mode_and_versioned_placeholders(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    with TestClient(create_app(config_path=str(path))) as client:
        assert client.get("/health").json()["masking_mode"] == "versioned"
        body = client.post("/mask", json={"text": "Tan Wei Ming, tan wei ming", "return_mapping": True}).json()
        assert body["text"] == "[PERSON_01_1], [PERSON_01_2]"
        assert body["mapping"] == {"[PERSON_01_1]": "Tan Wei Ming", "[PERSON_01_2]": "tan wei ming"}
