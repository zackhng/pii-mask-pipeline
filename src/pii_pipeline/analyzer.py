"""Presidio AnalyzerEngine assembled from the pipeline config."""
from __future__ import annotations

import copy

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NoOpNlpEngine

from .config import PipelineConfig
from .recognizers.deterministic import deny_list_recognizers, pattern_recognizers
from .recognizers.table import TableFieldRecognizer
from .registry import load_class
from .tables import load_table_fields, table_fields_from_sections


def source_of(result: RecognizerResult) -> str:
    """'deny_list' | 'patterns' | 'ner', from the recognizer name prefix."""
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


def _ner_recognizers(cfg: PipelineConfig) -> list:
    if cfg.ner is None:
        return []
    cls = load_class(cfg.ner.class_path)
    params = dict(cfg.ner.params)
    # label_map / min_score are forwarded only when set, so a custom NER class need not accept them
    if cfg.ner.label_map:
        params.setdefault("label_map", cfg.ner.label_map)
    if cfg.ner.min_score:
        params.setdefault("min_score", cfg.ner.min_score)
    recognizer = cls(**params)
    recognizer.name = f"ner:{recognizer.name}"
    return [recognizer]


def _table_recognizers(cfg: PipelineConfig) -> list:
    if cfg.tables is None:
        return []
    t = cfg.tables
    fields = (load_table_fields(t.fields_file) if t.fields_file
              else table_fields_from_sections(t.column_headers, t.row_headers))
    return [TableFieldRecognizer(fields, mode=mode, score=t.score) for mode in ("columns", "rows")]


def build_analyzer(cfg: PipelineConfig) -> AnalyzerEngine:
    base = [
        *deny_list_recognizers(cfg.deny_list),
        *_table_recognizers(cfg),
        *pattern_recognizers(cfg.patterns),
        *_ner_recognizers(cfg),
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
