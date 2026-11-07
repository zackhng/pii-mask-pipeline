"""PIIAnonymizer and OptimizedAnonymizer, driven directly with spans (no analyzer)."""
import pytest

from pii_pipeline.config import AnonymizerConfig, NerConfig, parse_config
from pii_pipeline.core.errors import ChecksumMismatchError
from pii_pipeline.core.types import DetectionResult, PIISpan, checksum_of
from pii_pipeline.masking.anonymizer import PIIAnonymizer
from pii_pipeline.masking.optimized_anonymizer import OptimizedAnonymizer, name_variants
from pii_pipeline.pipeline import PIIPipeline


def detect(text, found):
    """DetectionResult for exact phrases: {phrase: entity}; every occurrence is a span."""
    spans = []
    for phrase, entity in found.items():
        start = text.find(phrase)
        while start != -1:
            spans.append(PIISpan(start, start + len(phrase), phrase, entity, 0.9, "ner:Fake"))
            start = text.find(phrase, start + 1)
    spans.sort(key=lambda s: s.start)
    return DetectionResult(text, checksum_of(text), spans)


def anon(cls=PIIAnonymizer, **cfg):
    return cls(config=AnonymizerConfig(**cfg))


# ---------- surface propagation ----------

def test_known_entities_are_masked_everywhere_they_repeat():
    text = "Tan Wei Ming called. Later TAN WEI MING and tan  wei ming wrote."
    a = anon(entities_to_look_out_for=["PERSON"])
    assert a.anonymize(detect(text, {"Tan Wei Ming": "PERSON"}), text) == \
        "[PERSON_01] called. Later [PERSON_01] and [PERSON_01] wrote."
    assert [s.source for s in a.selected_spans] == ["ner", "propagated", "propagated"]


def test_without_look_out_for_only_detected_spans_are_masked():
    text = "Tan Wei Ming called. Later TAN WEI MING wrote."
    assert anon().anonymize(detect(text, {"Tan Wei Ming": "PERSON"}), text) == \
        "[PERSON_01] called. Later TAN WEI MING wrote."


def test_propagation_finds_split_emails_and_reformatted_accounts():
    text = "Write to john.doe@example.com (or john.doe@exam\nple.com). Acct 123-456 and 123456."
    a = anon(entities_to_look_out_for=["EMAIL", "ACCOUNT"], separator_insensitive=["ACCOUNT"])
    out = a.anonymize(detect(text, {"john.doe@example.com": "EMAIL", "123-456": "ACCOUNT"}), text)
    assert out == "Write to [EMAIL_01] (or [EMAIL_01]). Acct [ACCOUNT_01] and [ACCOUNT_01]."


def test_propagation_skips_word_fragments_and_excluded_values():
    text = "Lee met Leeson. Lee left. Our Firm and our firm."
    spans = [PIISpan(0, 3, "Lee", "PERSON", 0.9, "ner:Fake"), PIISpan(26, 34, "Our Firm", "BUSINESS", 0.9, "ner:Fake")]
    a = PIIAnonymizer(config=AnonymizerConfig(entities_to_look_out_for=["PERSON", "BUSINESS"]),
                      excluded_values=["our firm"])
    out = a.anonymize(DetectionResult(text, checksum_of(text), spans), text)
    assert out == "[PERSON_01] met Leeson. [PERSON_01] left. [BUSINESS_01] and our firm."


# ---------- mappings ----------

def test_mappings_returns_surfaces_and_filters_by_text():
    text = "Bob and bob"
    a = anon(entities_to_look_out_for=["PERSON"])
    a.anonymize(detect(text, {"Bob": "PERSON"}), text)
    assert a.mappings() == {"Bob": "[PERSON_01]", "bob": "[PERSON_01]"}
    assert a.mappings("only Bob here") == {"Bob": "[PERSON_01]"}
    assert a.token_mapping() == {"[PERSON_01]": "Bob"}


def test_load_mappings_reuses_user_tokens_and_reserves_their_numbers():
    a = anon()
    a.load_mappings({"Tan Wei Ming": "[PERSON_07]", "Acme": "<<client firm>>"})
    text = "Tan Wei Ming, TAN WEI MING, Bob and Acme"
    out = a.anonymize(detect(text, {"Tan Wei Ming": "PERSON", "TAN WEI MING": "PERSON", "Bob": "PERSON",
                                    "Acme": "BUSINESS"}), text)
    assert out == "[PERSON_07], [PERSON_07], [PERSON_08] and <<client firm>>"


def test_instance_keeps_tokens_consistent_across_documents():
    a = anon()
    a.anonymize(detect("Bob", {"Bob": "PERSON"}), "Bob")
    assert a.anonymize(detect("Ann and Bob", {"Ann": "PERSON", "Bob": "PERSON"}), "Ann and Bob") == \
        "[PERSON_02] and [PERSON_01]"
    a.reset()
    assert a.anonymize(detect("Ann", {"Ann": "PERSON"}), "Ann") == "[PERSON_01]"


# ---------- resolve ----------

def test_adjacent_spans_of_one_identity_merge():
    text = "Bob Bob\tBob, Bob"
    a = anon()
    out = a.anonymize(detect(text, {"Bob": "PERSON"}), text)
    assert out == "[PERSON_01], [PERSON_01]"
    assert [(s.start, s.end) for s in a.selected_spans] == [(0, 11), (13, 16)]


def test_adjacent_spans_of_different_people_do_not_merge():
    text = "Bob Ann"
    assert anon().anonymize(detect(text, {"Bob": "PERSON", "Ann": "PERSON"}), text) == "[PERSON_01] [PERSON_02]"


def test_checksum_mismatch_is_refused():
    result = detect("Bob", {"Bob": "PERSON"})
    with pytest.raises(ChecksumMismatchError):
        anon().resolve(result, "Rob")


def test_excluded_and_included_entities_and_urls():
    text = "Bob at Acme, see https://x.com/bob-tan and Bob"
    found = detect(text, {"Bob": "PERSON", "Acme": "BUSINESS", "bob-tan": "PERSON"})
    assert anon(included_entities=["BUSINESS"]).anonymize(found, text) == \
        "Bob at [BUSINESS_01], see https://x.com/bob-tan and Bob"
    assert PIIAnonymizer(config=AnonymizerConfig(), excluded_values=["Acme"], unmask_url=True).anonymize(found, text) == \
        "[PERSON_01] at Acme, see https://x.com/bob-tan and [PERSON_01]"


def test_resolve_does_not_build_text_and_sets_tokens():
    text = "Bob"
    spans = anon(mode="versioned").resolve(detect(text, {"Bob": "PERSON"}), text)
    assert [(s.token, s.text) for s in spans] == [("[PERSON_01_1]", "Bob")]


def test_per_entity_operator_runs_through_presidio():
    text = "Bob on 2020-01-01"
    a = anon(per_entity={"DOB": {"operator": "replace", "new_value": "<date>"}})
    assert a.anonymize(detect(text, {"Bob": "PERSON", "2020-01-01": "DOB"}), text) == "[PERSON_01] on <date>"


# ---------- optimized anonymizer ----------

def test_name_variants():
    v = name_variants("Tan Wei Ming")
    for form in ("Wei Ming", "Tan Wei", "Tan", "Wei", "Ming", "W. M. Tan", "W.M. Tan", "Mr Tan", "Ms Tan"):
        assert form in v
    assert "Tan Wei Ming" not in v
    assert name_variants("陈伟明") == [] and name_variants("Bob") == []
    assert v == sorted(v, key=lambda s: (-len(s), s))


def test_short_forms_of_a_detected_name_share_its_identity():
    text = "Tan Wei Ming opened the account. Mr Tan later said Wei Ming agreed. W. M. Tan signed."
    a = anon(OptimizedAnonymizer)
    out = a.anonymize(detect(text, {"Tan Wei Ming": "PERSON"}), text)
    assert out == "[PERSON_01] opened the account. [PERSON_01] later said [PERSON_01] agreed. [PERSON_01] signed."


def test_partial_form_next_to_another_name_is_not_masked():
    text = "Tan Wei Ming met Tan Ah Kow."
    out = anon(OptimizedAnonymizer).anonymize(detect(text, {"Tan Wei Ming": "PERSON"}), text)
    assert out == "[PERSON_01] met Tan Ah Kow."


def test_form_shared_by_two_detected_people_is_skipped():
    text = "Tan Wei Ming and Tan Ah Kow arrived. Tan spoke first."
    out = anon(OptimizedAnonymizer).anonymize(
        detect(text, {"Tan Wei Ming": "PERSON", "Tan Ah Kow": "PERSON"}), text)
    assert out == "[PERSON_01] and [PERSON_02] arrived. Tan spoke first."


def test_sentence_initial_capital_is_not_a_foreign_name():
    # "Yesterday" starts the sentence, so it does not count as another name before "Wei Ming"
    text = "Tan Wei Ming called. Yesterday Wei Ming confirmed."
    out = anon(OptimizedAnonymizer).anonymize(detect(text, {"Tan Wei Ming": "PERSON"}), text)
    assert out == "[PERSON_01] called. Yesterday [PERSON_01] confirmed."


def test_optimized_is_enabled_from_config():
    cfg = parse_config({"anonymizer": {"optimized": True}})
    cfg.ner = NerConfig(class_path="fake_ner:FakeNER", params={"phrases": {"Tan Wei Ming": "PERSON"}})
    assert PIIPipeline(cfg).mask("Tan Wei Ming. Mr Tan agreed.").text == "[PERSON_01]. [PERSON_01] agreed."
