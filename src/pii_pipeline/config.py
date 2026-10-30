"""YAML -> validated pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .checks import make_rule

SOURCES = ("deny_list", "table_columns", "table_rows", "patterns", "ner")


@dataclass
class NerConfig:
    class_path: str
    params: dict[str, Any] = field(default_factory=dict)
    label_map: dict[str, str] = field(default_factory=dict)
    min_score: float = 0.0


@dataclass
class DenyListItem:
    entity: str
    values: list[str]
    case_sensitive: bool = False


@dataclass
class PatternItem:
    name: str
    entity: str
    regex: str
    score: float = 0.5


@dataclass
class CheckItem:
    entity: str
    rule: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableConfig:
    # inline definitions (the usual case) or a separate YAML file; not both
    column_headers: dict[str, list[str]] = field(default_factory=dict)
    row_headers: dict[str, list[str]] = field(default_factory=dict)
    fields_file: str | None = None
    score: float = 0.9


MASKING_MODES = ("original", "versioned")


@dataclass
class MaskingConfig:
    format: str = "[{entity}_{index:02d}]"
    per_entity: dict[str, dict[str, Any]] = field(default_factory=dict)
    mode: str = "original"    # original | versioned; see masking.py
    # labels whose surface form varies only by separators ("123-456 789" == "123456789")
    separator_insensitive: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    languages: list[str]
    default_language: str
    ner: NerConfig | None
    deny_list: list[DenyListItem]
    patterns: list[PatternItem]
    allow_list: list[str]
    priority: list[str]
    masking: MaskingConfig
    tables: TableConfig | None = None
    checks: list[CheckItem] = field(default_factory=list)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid config: {msg}")


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    languages = raw.get("languages") or ["en"]
    _require(isinstance(languages, list) and all(isinstance(x, str) for x in languages), "languages must be a list of strings")
    default_language = raw.get("default_language", languages[0])
    _require(default_language in languages, "default_language must be one of languages")

    ner_raw = raw.get("ner")
    ner = None
    if ner_raw:
        _require("class" in ner_raw, "ner.class is required")
        ner = NerConfig(
            class_path=ner_raw["class"],
            params=dict(ner_raw.get("params") or {}),
            label_map=dict(ner_raw.get("label_map") or {}),
            min_score=float(ner_raw.get("min_score", 0.0)),
        )

    deny_list = []
    for item in raw.get("deny_list") or []:
        values = item.get("values") or []
        _require(bool(values), f"deny_list entry for {item.get('entity')} has no values")
        deny_list.append(DenyListItem(entity=item["entity"], values=[str(v) for v in values],
                                      case_sensitive=bool(item.get("case_sensitive", False))))

    patterns = []
    for item in raw.get("patterns") or []:
        score = float(item.get("score", 0.5))
        _require(0.0 <= score <= 1.0, f"pattern {item.get('name')} score must be in [0, 1]")
        patterns.append(PatternItem(name=item["name"], entity=item["entity"], regex=item["regex"], score=score))

    priority = list(raw.get("priority") or SOURCES)
    _require(sorted(priority) == sorted(SOURCES), f"priority must be a permutation of {list(SOURCES)}")

    m_raw = raw.get("masking") or {}
    mode = m_raw.get("mode", "original")
    _require(mode in MASKING_MODES, f"masking.mode must be one of {list(MASKING_MODES)}")
    default_format = "[{entity}_{index:02d}.{version}]" if mode == "versioned" else "[{entity}_{index:02d}]"
    fmt = m_raw.get("format", default_format)
    # separator_insensitive is a domain rule, so it is a top-level key next to ner / tables
    masking = MaskingConfig(format=fmt,
                            per_entity=dict(m_raw.get("per_entity") or {}),
                            mode=mode,
                            separator_insensitive=[str(x) for x in raw.get("separator_insensitive") or []])
    _require("{entity}" in fmt and "{index" in fmt, "masking.format must contain {entity} and {index}")
    if mode == "versioned":
        _require("{version" in fmt, "masking.format must contain {version} in versioned mode")

    tables = None
    t_raw = raw.get("tables")
    if t_raw:
        has_file = "fields_file" in t_raw
        has_inline = "column_headers" in t_raw or "row_headers" in t_raw
        _require(has_file or has_inline, "tables: give fields_file, or column_headers / row_headers")
        _require(not (has_file and has_inline), "tables: give either fields_file or column_headers/row_headers, not both")
        t_score = float(t_raw.get("score", 0.9))
        _require(0.0 <= t_score <= 1.0, "tables.score must be in [0, 1]")
        tables = TableConfig(
            column_headers=dict(t_raw.get("column_headers") or {}),
            row_headers=dict(t_raw.get("row_headers") or {}),
            fields_file=t_raw.get("fields_file"),
            score=t_score,
        )

    # checks: {LABEL: [{rule: name, <params>}, ...]}. Validated here so a bad rule fails at load time.
    checks = []
    for entity, items in (raw.get("checks") or {}).items():
        for item in items or []:
            _require("rule" in item, f"checks.{entity} entry needs a rule name")
            params = {k: v for k, v in item.items() if k != "rule"}
            try:
                make_rule(item["rule"], params)
            except (ValueError, ImportError, AttributeError) as exc:
                raise ValueError(f"invalid config: checks.{entity} {item['rule']}: {exc}") from exc
            checks.append(CheckItem(entity=str(entity), rule=item["rule"], params=params))

    return PipelineConfig(
        languages=languages,
        default_language=default_language,
        ner=ner,
        deny_list=deny_list,
        patterns=patterns,
        allow_list=[str(x) for x in raw.get("allow_list") or []],
        priority=priority,
        masking=masking,
        tables=tables,
        checks=checks,
    )


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f) or {})
