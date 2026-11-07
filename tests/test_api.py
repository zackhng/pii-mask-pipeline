"""HTTP API tests. Use a temporary config with the fake NER, so no model is loaded."""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from pii_pipeline.api import create_app

TODAY = date(2025, 10, 30)

CONFIG = """
languages: [en, zh]
default_language: en
ner:
  class: fake_ner:FakeNER
  params:
    phrases:
      "Tan Wei Ming": PERSON
      "2020-05-01": DOB
      "123-456789-001": ACCOUNT
anonymizer:
  allow_listed_values: ["Our Firm Pte Ltd"]
  span_rules:
    DOB:
      - {rule: "check_rules:dob_adult", years: 18}
"""


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    app = create_app(config_path=str(path), max_text_chars=200, today=TODAY)
    with TestClient(app) as c:
        yield c


def test_health_reports_config(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "languages": ["en", "zh"], "default_language": "en",
                    "ner": "fake_ner:FakeNER", "checks": 1, "masking_mode": "original"}


def test_analyze_returns_spans_without_text_or_masking(client):
    r = client.post("/analyze", json={"text": "Tan Wei Ming, account 123-456789-001"})
    assert r.status_code == 200
    body = r.json()
    assert [(e["entity_type"], e["start"], e["end"]) for e in body["entities"]] == [
        ("PERSON", 0, 12), ("ACCOUNT", 22, 36)]
    assert all("placeholder" not in e and "text" not in e for e in body["entities"])
    assert body["rejected"] == []


def test_analyze_reports_gate_rejections(client):
    body = client.post("/analyze", json={"text": "born 2020-05-01"}).json()
    assert body["entities"] == []
    assert [(r["entity_type"], r["rule"]) for r in body["rejected"]] == [("DOB", "check_rules:dob_adult")]


def test_mask_returns_masked_text_and_placeholders(client):
    r = client.post("/mask", json={"text": "Tan Wei Ming, account 123-456789-001"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "[PERSON_01], account [ACCOUNT_01]"
    assert [e["placeholder"] for e in body["entities"]] == ["[PERSON_01]", "[ACCOUNT_01]"]
    assert body["mapping"] is None


def test_mask_mapping_only_when_requested(client):
    body = client.post("/mask", json={"text": "Tan Wei Ming", "return_mapping": True}).json()
    assert body["mapping"] == {"[PERSON_01]": "Tan Wei Ming"}


def test_mask_respects_allow_list_and_gate(client):
    body = client.post("/mask", json={"text": "Our Firm Pte Ltd, born 2020-05-01"}).json()
    assert body["text"] == "Our Firm Pte Ltd, born 2020-05-01"


def test_unknown_language_is_422(client):
    r = client.post("/mask", json={"text": "Tan Wei Ming", "language": "fr"})
    assert r.status_code == 422
    assert "fr" in r.json()["detail"]


def test_configured_language_is_accepted(client):
    assert client.post("/analyze", json={"text": "Tan Wei Ming", "language": "zh"}).status_code == 200


def test_text_over_limit_is_413(client):
    r = client.post("/mask", json={"text": "x" * 201})
    assert r.status_code == 413


def test_missing_text_is_422(client):
    assert client.post("/mask", json={}).status_code == 422


def test_empty_text_is_fine(client):
    body = client.post("/mask", json={"text": ""}).json()
    assert body == {"text": "", "entities": [], "rejected": [], "mapping": None}
