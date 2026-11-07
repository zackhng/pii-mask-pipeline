"""Presidio AnalyzerEngine assembled from the detector config."""
from __future__ import annotations

import copy

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NoOpNlpEngine

from .config.detector_config import DetectorConfig
from .core.errors import ConfigError, ModelNotAvailableError, PIIError
from .recognizers.deterministic import deny_list_recognizers, pattern_recognizers, presidio_builtin_recognizers
from .recognizers.table import StructuredLabelRecognizer
from .registry import load_class


def source_of(result: RecognizerResult) -> str:
    """The detector family, from the recognizer name prefix: 'deny_list', 'patterns', 'ner', ..."""
    name = (result.recognition_metadata or {}).get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
    return name.split(":", 1)[0]


def merge_overlaps(results: list[RecognizerResult], priority: list[str]) -> list[RecognizerResult]:
    """Merge overlapping spans into one span covering their union, so no part of a detected entity
    stays visible. Transitive: a-b and b-c overlapping gives one span a-c. Spans that only touch
    (end == start) are not merged.

    The merged span takes the label of its best member, ordered by source priority, then score,
    then longer span, then position. Presidio only drops same-type containment, so cross-type
    overlaps need this step."""
    rank = {s: i for i, s in enumerate(priority)}

    def best_key(r: RecognizerResult):
        return (rank.get(source_of(r), len(rank)), -r.score, -(r.end - r.start), r.start)

    merged: list[RecognizerResult] = []
    cluster: list[RecognizerResult] = []
    cluster_end = -1

    def flush():
        if not cluster:
            return
        best = min(cluster, key=best_key)
        merged.append(RecognizerResult(
            entity_type=best.entity_type,
            start=min(r.start for r in cluster),
            end=max(r.end for r in cluster),
            score=best.score,
            recognition_metadata=best.recognition_metadata,
        ))

    for r in sorted(results, key=lambda r: (r.start, r.end)):
        if cluster and r.start < cluster_end:
            cluster.append(r)
            cluster_end = max(cluster_end, r.end)
        else:
            flush()
            cluster = [r]
            cluster_end = r.end
    flush()
    return merged


def build_ner(cfg: DetectorConfig) -> list:
    """The configured NER recognizer, or nothing. Load failures raise ModelNotAvailableError."""
    if cfg.ner is None:
        return []
    ner = cfg.ner
    try:
        cls = load_class(ner.class_path)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"ner class {ner.class_path!r} cannot be imported: {exc}") from exc
    params = dict(ner.params)
    # for custom classes label_map / min_score are forwarded only when set, so a class need not accept them
    if ner.label_map:
        params.setdefault("label_map", ner.label_map)
    if ner.min_score:
        params.setdefault("min_score", ner.min_score)
    try:
        recognizer = cls(**params)
    except PIIError:
        raise
    except TypeError as exc:
        raise ConfigError(f"ner {ner.display_name}: bad settings: {exc}") from exc
    except (OSError, ImportError) as exc:
        raise ModelNotAvailableError(f"ner {ner.display_name} could not be loaded: {exc}") from exc
    recognizer.name = f"ner:{recognizer.name}"
    return [recognizer]


def _structured_recognizers(cfg: DetectorConfig) -> list:
    a = cfg.anonymizer
    out = []
    if a.table_semantics is not None:
        fields = a.table_semantics.fields()
        out += [StructuredLabelRecognizer(mode, score=a.table_semantics.score, fields=fields)
                for mode in ("columns", "rows")]
    aliases = a.inline_aliases()
    if aliases:
        out.append(StructuredLabelRecognizer("inline", score=a.inline_score, aliases=aliases))
    return out


def build_analyzer(cfg: DetectorConfig) -> AnalyzerEngine:
    base = [
        *deny_list_recognizers(cfg.presidio_recognizers.deny_list),
        *_structured_recognizers(cfg),
        *pattern_recognizers(cfg.patterns),
        *presidio_builtin_recognizers(cfg.presidio_recognizers.builtin),
        *build_ner(cfg),
    ]
    registry = RecognizerRegistry(supported_languages=cfg.languages)
    for lang in cfg.languages:
        for rec in base:
            # shallow copy per language: the NER model is shared, not reloaded
            clone = copy.copy(rec)
            clone.supported_language = lang
            registry.add_recognizer(clone)

    nlp = NoOpNlpEngine(models=[{"lang_code": lang, "model_name": "none"} for lang in cfg.languages])
    return AnalyzerEngine(registry=registry, nlp_engine=nlp, supported_languages=cfg.languages)
