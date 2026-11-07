"""GLiNER adapter (with a fake `gliner` package) and detector-config loading."""
import sys
import types

import pytest

from pii_pipeline.config import load_config, parse_config
from pii_pipeline.core.errors import ConfigError, ModelNotAvailableError
from pii_pipeline.pipeline import PIIPipeline
from pii_pipeline.recognizers.gliner import GLiNERRecognizer, windows


class FakeGLiNER:
    """Finds fixed phrases per prompt label, like GLiNER.predict_entities."""
    loaded: list = []
    PHRASES = {"person": ["Tan Wei Ming"], "phone number": ["+65 6123 4567"]}

    @classmethod
    def from_pretrained(cls, name):
        cls.loaded.append(name)
        return cls()

    def to(self, device):
        return self

    def predict_entities(self, text, labels, threshold=0.5, flat_ner=True):
        out = []
        for label in labels:
            for phrase in self.PHRASES.get(label, []):
                i = text.find(phrase)
                while i != -1:
                    out.append({"start": i, "end": i + len(phrase), "text": phrase, "label": label, "score": 0.91})
                    i = text.find(phrase, i + 1)
        return out


@pytest.fixture
def fake_gliner(monkeypatch):
    FakeGLiNER.loaded = []
    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=FakeGLiNER))
    return FakeGLiNER


# ---------- windows ----------

def test_windows_cover_text_with_overlap_and_cut_at_spaces():
    text = " ".join(f"word{i}" for i in range(200))
    wins = windows(text, 100, 20)
    assert wins[0][0] == 0 and wins[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(wins, wins[1:]):
        assert s2 < e1 and s2 > s1                    # overlap and progress
        assert e1 == len(text) or text[e1] == " "      # cuts at whitespace
    assert windows("short", 100, 20) == [(0, 5)]


def test_windows_handle_text_without_spaces():
    wins = windows("陈" * 250, 100, 10)
    assert wins[0] == (0, 100) and wins[-1][1] == 250


# ---------- adapter ----------

def test_adapter_maps_labels_and_offsets_across_windows(fake_gliner):
    rec = GLiNERRecognizer(model="fake/model", labels={"person": "PERSON", "phone number": "PHONE"},
                           window_size=60, stride=30)
    text = ("filler " * 12) + "Tan Wei Ming, call +65 6123 4567. " + ("more " * 10) + "Tan Wei Ming"
    results = rec.analyze(text, entities=None)
    found = sorted((r.entity_type, text[r.start:r.end]) for r in results)
    assert found == [("PERSON", "Tan Wei Ming"), ("PERSON", "Tan Wei Ming"), ("PHONE", "+65 6123 4567")]
    assert fake_gliner.loaded == ["fake/model"]


def test_adapter_list_labels_and_label_map(fake_gliner):
    rec = GLiNERRecognizer(model="m", labels=["person"], label_map={"PERSON": "NAME"})
    assert rec.supported_entities == ["NAME"]
    assert [r.entity_type for r in rec.analyze("Hi Tan Wei Ming", None)] == ["NAME"]


def test_missing_gliner_package_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "gliner", None)   # makes `import gliner` fail
    with pytest.raises(ModelNotAvailableError, match=r"pip install \.\[gliner\]"):
        GLiNERRecognizer(model="m", labels=["person"])


def test_pipeline_runs_a_gliner_model_chosen_by_name(fake_gliner):
    cfg = parse_config({
        "ner": {"model": "g", "threshold": 0.4},
        "model": {"gliner": {"g": {"model": "fake/g", "labels": {"person": "PERSON"}, "threshold": 0.9}}},
    })
    assert cfg.ner.backend == "gliner" and cfg.ner.params["threshold"] == 0.4   # ner settings win
    res = PIIPipeline(cfg).mask("Hi Tan Wei Ming")
    assert res.text == "Hi [PERSON_01]" and res.entities[0]["source"] == "ner"


# ---------- config ----------

def test_shipped_config_loads_with_three_models():
    cfg = load_config()
    assert cfg.ner.model == "xlmr_v5" and cfg.ner.backend == "hf_bio"
    assert cfg.ner.params["max_length"] == 384                  # window_size -> max_length for BIO models
    assert set(cfg.models) == {"xlmr_v5", "gliner_multi_pii", "gliner_small"}
    assert cfg.patterns == []                                   # regex_patterns.yaml ships examples only
    assert cfg.anonymizer.table_semantics is not None and cfg.anonymizer.span_alias


def test_regex_patterns_file_is_relative_to_the_config(tmp_path):
    (tmp_path / "pats.yaml").write_text(
        "- {label: TIN, name: nric, pattern: '\\b[STFG]\\d{7}[A-Z]\\b', score: 0.85}\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("regex_patterns_file: pats.yaml\n", encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    assert [(p.name, p.entity) for p in cfg.patterns] == [("nric", "TIN")]
    assert PIIPipeline(cfg).mask("NRIC S1234567D").text == "NRIC [TIN_01]"


def test_regex_context_words(tmp_path):
    cfg = parse_config({"regex_patterns": [
        {"label": "PHONE", "name": "ph", "pattern": r"\b\d{8}\b", "context": ["tel", "phone"], "context_window": 6}]})
    assert PIIPipeline(cfg).mask("Tel 61234567, order 12345678").text == "Tel [PHONE_01], order 12345678"


def test_presidio_builtin_recognizers():
    cfg = parse_config({"presidio_recognizers": {"builtin": ["EmailRecognizer"]}})
    res = PIIPipeline(cfg).mask("mail bob@example.com")
    assert res.text == "mail [EMAIL_ADDRESS_01]" and res.entities[0]["source"] == "presidio"
    with pytest.raises(ConfigError, match="NotARecognizer"):
        PIIPipeline(parse_config({"presidio_recognizers": {"builtin": ["NotARecognizer"]}}))


@pytest.mark.parametrize("raw, message", [
    ({"ner": {"model": "nope"}}, "not defined under model"),
    ({"ner": {"threshold": 0.5}}, "needs a model"),
    ({"model": {"weird": {"x": {}}}}, "unknown section"),
    ({"model": {"gliner": {"g": {"labels": ["person"]}}}, "ner": {"model": "g"}}, "needs model"),
    ({"regex_patterns": [{"label": "X", "name": "x"}]}, "needs 'pattern'"),
    ({"anonymizer": {"span_variant_rules": {"EMAIL": ["sparkles"]}}}, "unknown rules"),
    ({"anonymizer": {"per_entity": {"DOB": {"new_value": "x"}}}}, "needs an operator"),
    ({"surprise": 1}, "unknown key"),
])
def test_config_errors(raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_missing_config_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
