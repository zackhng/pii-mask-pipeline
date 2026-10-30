"""Indexed placeholder masking on top of Presidio's AnonymizerEngine.

Presidio applies custom operators in reverse text order, so placeholders are assigned up front, walking
the results in ascending position.

Two modes (masking.mode):

original   One placeholder per value. Surface variants of one value (case, spacing, and separators for
           labels in `separator_insensitive`) share an index and a placeholder. The mapping keeps only
           the first variant seen, so it is not a 1:1 map back to the text.
           Example: [PERSON_01] for both "Tan Wei Ming" and "tan wei ming".

versioned  One placeholder per exact surface form. Variants of a value share an index, and each distinct
           exact text gets its own version within that index. Every placeholder maps back to exactly one
           original string, and every original string maps to exactly one placeholder.
           Example: [PERSON_01.1] for "Tan Wei Ming", [PERSON_01.2] for "tan wei ming".
"""
from __future__ import annotations

import re

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .config import MaskingConfig


def normalise(text: str, entity: str, separator_insensitive: frozenset = frozenset()) -> str:
    """Key used to decide whether two surface forms are the same value. Case and whitespace are
    ignored everywhere; separators are ignored only for labels in `separator_insensitive`
    (e.g. "123-456 789" == "123456789" for ACCOUNT)."""
    value = " ".join(text.casefold().split())
    if entity in separator_insensitive:
        value = re.sub(r"[\s.\-]", "", value)
    return value


def _lookup_key(text: str, entity: str, mode: str, sep: frozenset) -> str:
    """What decides a placeholder: the exact text (versioned) or the normalised value (original)."""
    return text if mode == "versioned" else normalise(text, entity, sep)


def assign_placeholders(text: str, results: list[RecognizerResult], cfg: MaskingConfig) -> list[str]:
    """Placeholder for each result, in the order of `results`. `results` must be non-overlapping."""
    sep = frozenset(cfg.separator_insensitive)
    counters: dict[str, int] = {}
    index: dict[tuple[str, str], int] = {}          # (entity, normalised value) -> index
    versions: dict[tuple[str, str], dict[str, int]] = {}   # (entity, normalised value) -> exact text -> version
    out: list[str | None] = [None] * len(results)

    for i in sorted(range(len(results)), key=lambda i: results[i].start):
        r = results[i]
        exact = text[r.start:r.end]
        key = (r.entity_type, normalise(exact, r.entity_type, sep))
        if key not in index:
            counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
            index[key] = counters[r.entity_type]
            versions[key] = {}
        version = 1
        if cfg.mode == "versioned":
            seen = versions[key]
            if exact not in seen:
                seen[exact] = len(seen) + 1
            version = seen[exact]
        out[i] = cfg.format.format(entity=r.entity_type, index=index[key], version=version)
    return out


def mask_text(text: str, results: list[RecognizerResult], cfg: MaskingConfig,
              engine: AnonymizerEngine | None = None, return_mapping: bool = False):
    """`results` must be non-overlapping (see analyzer.merge_overlaps).
    Returns (masked_text, placeholder -> original mapping or None, placeholder per result, in order)."""
    engine = engine or AnonymizerEngine()
    sep = frozenset(cfg.separator_insensitive)
    placeholders = assign_placeholders(text, results, cfg)

    # Presidio calls the operator with the matched text, so look placeholders up by the same key.
    table: dict[tuple[str, str], str] = {}
    for r, ph in zip(results, placeholders):
        exact = text[r.start:r.end]
        table.setdefault((r.entity_type, _lookup_key(exact, r.entity_type, cfg.mode, sep)), ph)

    operators: dict[str, OperatorConfig] = {}
    for entity in {r.entity_type for r in results}:
        if entity in cfg.per_entity:
            spec = dict(cfg.per_entity[entity])
            operators[entity] = OperatorConfig(spec.pop("operator"), spec)
            continue

        def placeholder(original: str, entity=entity) -> str:
            return table[(entity, _lookup_key(original, entity, cfg.mode, sep))]

        operators[entity] = OperatorConfig("custom", {"lambda": placeholder})

    anonymized = engine.anonymize(text=text, analyzer_results=results, operators=operators)

    mapping = None
    if return_mapping:
        mapping = {}
        for r, ph in zip(results, placeholders):
            mapping.setdefault(ph, text[r.start:r.end])
    return anonymized.text, mapping, placeholders


def unmask(masked: str, mapping: dict[str, str]) -> str:
    """Restore original text from masked text and a placeholder -> original mapping. Single pass, so
    text inserted by one replacement is never replaced again."""
    if not mapping:
        return masked
    pattern = re.compile("|".join(re.escape(p) for p in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group(0)], masked)
