"""Real-model tests for the v5.0 NER recognizer. Skipped if the weights are not extracted to models/."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "xlmr-base-pii-ml-v5.0"
SIBLING_SRC = ROOT.parent / "token_classification_ft" / "src"

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(),
                                reason="v5.0 weights not extracted to models/")

MODEL_CARD_EXAMPLE = "Hi Tan Wei Ming (陈伟明), 已经 transfer SGD 25,000 到 account 123-456789-001, ref TT240918。"


@pytest.fixture(scope="module")
def pipeline():
    from pii_pipeline.config import parse_config, NerConfig
    from pii_pipeline.pipeline import PIIPipeline
    cfg = parse_config({"languages": ["en", "zh"], "default_language": "en"})
    cfg.ner = NerConfig(class_path="pii_pipeline.recognizers.ner_hf:HFTokenClassRecognizer",
                        params={"model_dir": str(MODEL_DIR), "device": "auto"})
    return PIIPipeline(cfg)


def test_model_card_example(pipeline):
    res = pipeline.mask(MODEL_CARD_EXAMPLE, return_mapping=True)
    types = [e["entity_type"] for e in res.entities]
    assert types.count("PERSON") == 2
    assert "ACCOUNT" in types
    assert "25,000" in res.text and "TT240918" in res.text   # amount and reference stay unlabelled
    assert "123-456789-001" not in res.text and "Tan Wei Ming" not in res.text
    assert res.mapping["[ACCOUNT_01]"] == "123-456789-001"


def test_deterministic_layer_on_top_of_ner(pipeline):
    # Deny list and NER run together in one analyzer: the deny-list value is masked even when the model misses it.
    from pii_pipeline.analyzer import build_analyzer
    from pii_pipeline.config import DenyListItem
    original = list(pipeline.config.deny_list)
    pipeline.config.deny_list = original + [DenyListItem(entity="BUSINESS", values=["Zebra Quartz Holdings"])]
    pipeline.analyzer = build_analyzer(pipeline.config)
    try:
        out = pipeline.mask("Funds from Zebra Quartz Holdings were sent to Tan Wei Ming.").text
        assert "Zebra Quartz Holdings" not in out
        assert "Tan Wei Ming" not in out
    finally:
        pipeline.config.deny_list = original
        pipeline.analyzer = build_analyzer(pipeline.config)


@pytest.mark.skipif(not (SIBLING_SRC / "predict.py").exists(), reason="sibling token_classification_ft not present")
def test_parity_with_sibling_predictor(pipeline):
    """Ported windowing + decode + trim must give the same spans as the sibling PIIPredictor."""
    sys.path.insert(0, str(SIBLING_SRC))
    try:
        from predict import PIIPredictor
    except Exception as exc:  # sibling deps missing in this env
        pytest.skip(f"sibling predictor not importable: {exc}")
    sibling = PIIPredictor(str(MODEL_DIR), device="cpu")
    docs = [
        MODEL_CARD_EXAMPLE,
        "Client: Lee Kuan Yew.\nDOB: 16/09/1923\nEmail: kuan@example.com, phone +65 6123 4567.",
        "Account 0123456789 at DBS, Acme Holdings Pte. Ltd., 10 Marina Blvd, Singapore 018983.",
    ]
    rec = pipeline.analyzer.registry.recognizers
    ner = next(r for r in rec if r.name.startswith("ner:") and r.supported_language == "en")
    for doc in docs:
        ours = sorted((s["start"], s["end"], s["label"]) for s in ner.predict_spans(doc))
        theirs = sorted((s["start"], s["end"], s["label"]) for s in sibling.predict(doc))
        assert ours == theirs, doc
