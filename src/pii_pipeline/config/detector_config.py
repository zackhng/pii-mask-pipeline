"""Load the detector configuration (detector_config.yaml + regex_patterns.yaml).

Detector backends are configured independently of the NLP engine. The analyzer runs on Presidio's
no-op NLP engine; every detector is a Presidio recognizer:

    presidio_recognizers   Presidio's predefined recognizers (by class name) and exact-value deny lists
    regex_patterns         regex -> label, from regex_patterns.yaml and/or inline
    anonymizer             what to mask and how: placeholders, allow list, aliases, tables, span rules
    ner                    the chosen model and its run-time settings
    model                  every available model: token_classification (BIO), gliner, custom

All validation errors raise ConfigError (a ValueError) with a message starting "invalid config:".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ConfigError
from ..masking.span_rules import make_rule
from ..masking.structured_rules import TableFields, invert_aliases, table_fields_from_sections
from ..masking.tokens import MODES, default_format, validate_format

SOURCES = ("deny_list", "table_columns", "table_rows", "inline_labels", "patterns", "presidio", "ner")

BACKENDS = {
    "hf_bio": "pii_pipeline.recognizers.ner_hf:HFTokenClassRecognizer",
    "gliner": "pii_pipeline.recognizers.gliner:GLiNERRecognizer",
}
MODEL_SECTIONS = {"token_classification": "hf_bio", "gliner": "gliner", "custom": "custom"}
# generic `ner` setting -> constructor argument, per backend
NER_SETTING_NAMES = {
    "hf_bio": {"window_size": "max_length", "threshold": "min_score"},
    "gliner": {},
}

TOP_LEVEL_KEYS = {"languages", "default_language", "regex_patterns_file", "regex_patterns",
                  "presidio_recognizers", "anonymizer", "ner", "model"}
MOVED_KEYS = {
    "allow_list": "anonymizer.allow_listed_values",
    "priority": "anonymizer.priority",
    "masking": "anonymizer (mode, format, per_entity)",
    "separator_insensitive": "anonymizer.separator_insensitive",
    "deny_list": "presidio_recognizers.deny_list",
    "patterns": "regex_patterns (or regex_patterns.yaml)",
    "tables": "anonymizer.table_semantics",
    "checks": "anonymizer.span_rules",
}


# ---------- dataclasses ----------

@dataclass
class NerConfig:
    """The resolved NER recognizer: a class and its constructor arguments."""
    class_path: str
    params: dict[str, Any] = field(default_factory=dict)
    label_map: dict[str, str] = field(default_factory=dict)   # forwarded to custom classes only when set
    min_score: float = 0.0                                    # forwarded to custom classes only when set
    model: str | None = None                                  # key under `model.*`, if chosen by name
    backend: str = "custom"                                   # hf_bio | gliner | custom

    @property
    def display_name(self) -> str:
        return self.model or self.class_path


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
    context: list[str] = field(default_factory=list)   # if set, a match needs one of these words nearby
    context_window: int = 40


@dataclass
class PresidioConfig:
    builtin: list[str] = field(default_factory=list)
    deny_list: list[DenyListItem] = field(default_factory=list)


@dataclass
class TableField:
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class TableSemantics:
    columns: list[TableField] = field(default_factory=list)
    rows: list[TableField] = field(default_factory=list)
    score: float = 0.9

    def fields(self) -> TableFields:
        def section(items):
            out: dict[str, list[str]] = {}
            for f in items:
                out.setdefault(f.entity_type, []).extend([f.name, *f.aliases])
            return out
        return table_fields_from_sections(section(self.columns), section(self.rows))


@dataclass
class SpanRuleItem:
    entity: str        # a label, or "*" for every label
    rule: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnonymizerConfig:
    mode: str = "original"
    format: str = ""
    per_entity: dict[str, dict[str, Any]] = field(default_factory=dict)
    priority: list[str] = field(default_factory=lambda: list(SOURCES))
    allow_listed_values: list[str] = field(default_factory=list)
    entities_to_look_out_for: list[str] = field(default_factory=list)
    separator_insensitive: list[str] = field(default_factory=list)
    span_variant_rules: dict[str, list[str]] = field(default_factory=dict)
    unmask_url: bool = False
    included_entities: list[str] | None = None
    optimized: bool = False
    name_entities: list[str] = field(default_factory=lambda: ["PERSON"])
    span_alias: dict[str, list[str]] = field(default_factory=dict)
    inline_score: float = 0.9
    table_semantics: TableSemantics | None = None
    span_rules: list[SpanRuleItem] = field(default_factory=list)

    def __post_init__(self):
        if not self.format and self.mode in MODES:
            self.format = default_format(self.mode)

    def inline_aliases(self) -> dict[str, str]:
        return invert_aliases(self.span_alias, "anonymizer.span_alias")


@dataclass
class DetectorConfig:
    languages: list[str] = field(default_factory=lambda: ["en"])
    default_language: str = "en"
    presidio_recognizers: PresidioConfig = field(default_factory=PresidioConfig)
    patterns: list[PatternItem] = field(default_factory=list)
    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    ner: NerConfig | None = None
    models: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> raw entry (with backend)


# ---------- parsing ----------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _str_list(value, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    _require(isinstance(value, list), f"{name} must be a list")
    return [str(v) for v in value]


def _score(value, name: str, default: float) -> float:
    score = float(default if value is None else value)
    _require(0.0 <= score <= 1.0, f"{name} must be in [0, 1]")
    return score


def _parse_patterns(items, where: str) -> list[PatternItem]:
    if items is None:
        return []
    if isinstance(items, dict):
        items = items.get("patterns") or []
    _require(isinstance(items, list), f"{where} must be a list of {{label, name, pattern}}")
    out = []
    for i, item in enumerate(items):
        _require(isinstance(item, dict), f"{where}[{i}] must be a mapping")
        for key in ("label", "name", "pattern"):
            _require(key in item, f"{where}[{i}] needs '{key}'")
        out.append(PatternItem(
            name=str(item["name"]), entity=str(item["label"]), regex=str(item["pattern"]),
            score=_score(item.get("score"), f"{where} {item['name']} score", 0.5),
            context=_str_list(item.get("context"), f"{where} {item['name']} context"),
            context_window=int(item.get("context_window", 40)),
        ))
    return out


def _load_pattern_file(path: str, base_dir: Path | None) -> list[PatternItem]:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    _require(p.is_file(), f"regex_patterns_file {path!r} not found")
    with open(p, encoding="utf-8") as f:
        return _parse_patterns(yaml.safe_load(f), f"regex_patterns_file {p.name}")


def _parse_presidio(raw) -> PresidioConfig:
    raw = raw or {}
    deny = []
    for item in raw.get("deny_list") or []:
        values = item.get("values") or []
        _require("entity" in item, "presidio_recognizers.deny_list entries need an entity")
        _require(bool(values), f"presidio_recognizers.deny_list entry for {item.get('entity')} has no values")
        deny.append(DenyListItem(entity=str(item["entity"]), values=[str(v) for v in values],
                                 case_sensitive=bool(item.get("case_sensitive", False))))
    return PresidioConfig(builtin=_str_list(raw.get("builtin"), "presidio_recognizers.builtin"), deny_list=deny)


def _parse_table_semantics(raw) -> TableSemantics | None:
    if not raw:
        return None

    def fields(items, side):
        out = []
        for i, item in enumerate(items or []):
            _require(isinstance(item, dict) and "entity_type" in item,
                     f"table_semantics.{side}[{i}] needs entity_type")
            name = str(item.get("name") or "")
            aliases = _str_list(item.get("aliases"), f"table_semantics.{side}[{i}].aliases")
            _require(bool(name or aliases), f"table_semantics.{side}[{i}] needs a name or aliases")
            out.append(TableField(name=name, entity_type=str(item["entity_type"]),
                                  aliases=[a for a in aliases if a]))
        return out

    ts = TableSemantics(columns=fields(raw.get("columns"), "columns"), rows=fields(raw.get("rows"), "rows"),
                        score=_score(raw.get("score"), "table_semantics.score", 0.9))
    _require(bool(ts.columns or ts.rows), "table_semantics needs columns or rows")
    try:
        ts.fields()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return ts


def _parse_span_rules(raw) -> list[SpanRuleItem]:
    out = []
    for entity, items in (raw or {}).items():
        for item in items or []:
            _require(isinstance(item, dict) and "rule" in item, f"span_rules.{entity} entry needs a rule name")
            params = {k: v for k, v in item.items() if k != "rule"}
            try:
                make_rule(str(item["rule"]), params)
            except (ValueError, ImportError, AttributeError) as exc:
                raise ConfigError(f"span_rules.{entity} {item['rule']}: {exc}") from exc
            out.append(SpanRuleItem(entity=str(entity), rule=str(item["rule"]), params=params))
    return out


def _parse_priority(raw) -> list[str]:
    if raw is None:
        return list(SOURCES)
    priority = _str_list(raw, "anonymizer.priority")
    unknown = [s for s in priority if s not in SOURCES]
    _require(not unknown, f"anonymizer.priority has unknown sources {unknown}; known: {list(SOURCES)}")
    _require(len(set(priority)) == len(priority), "anonymizer.priority lists a source twice")
    return priority + [s for s in SOURCES if s not in priority]   # unlisted sources rank last


def _parse_anonymizer(raw) -> AnonymizerConfig:
    raw = raw or {}
    mode = raw.get("mode", "original")
    _require(mode in MODES, f"anonymizer.mode must be one of {list(MODES)}")
    fmt = raw.get("format") or default_format(mode)
    try:
        validate_format(fmt, mode)
    except ValueError as exc:
        raise ConfigError(f"anonymizer.format: {exc}") from exc

    variant_rules = {str(k): _str_list(v, f"span_variant_rules.{k}") for k, v in (raw.get("span_variant_rules") or {}).items()}
    from ..masking.span_variants import RULES
    for label, rules in variant_rules.items():
        bad = [r for r in rules if r not in RULES]
        _require(not bad, f"anonymizer.span_variant_rules.{label}: unknown rules {bad}; known: {list(RULES)}")

    included = raw.get("included_entities")
    cfg = AnonymizerConfig(
        mode=mode,
        format=fmt,
        per_entity=dict(raw.get("per_entity") or {}),
        priority=_parse_priority(raw.get("priority")),
        allow_listed_values=_str_list(raw.get("allow_listed_values"), "anonymizer.allow_listed_values"),
        entities_to_look_out_for=_str_list(raw.get("entities_to_look_out_for"), "anonymizer.entities_to_look_out_for"),
        separator_insensitive=_str_list(raw.get("separator_insensitive"), "anonymizer.separator_insensitive"),
        span_variant_rules=variant_rules,
        unmask_url=bool(raw.get("unmask_url", False)),
        included_entities=None if included is None else _str_list(included, "anonymizer.included_entities"),
        optimized=bool(raw.get("optimized", False)),
        name_entities=_str_list(raw.get("name_entities", ["PERSON"]), "anonymizer.name_entities"),
        span_alias={str(k): _str_list(v, f"span_alias.{k}") for k, v in (raw.get("span_alias") or {}).items()},
        inline_score=_score(raw.get("inline_score"), "anonymizer.inline_score", 0.9),
        table_semantics=_parse_table_semantics(raw.get("table_semantics")),
        span_rules=_parse_span_rules(raw.get("span_rules")),
    )
    for entity, spec in cfg.per_entity.items():
        _require(isinstance(spec, dict) and "operator" in spec, f"anonymizer.per_entity.{entity} needs an operator")
    try:
        cfg.inline_aliases()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return cfg


def _parse_models(raw) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for section, entries in (raw or {}).items():
        _require(section in MODEL_SECTIONS, f"model.{section}: unknown section; use {list(MODEL_SECTIONS)}")
        for name, entry in (entries or {}).items():
            _require(isinstance(entry, dict), f"model.{section}.{name} must be a mapping")
            _require(name not in models, f"model name {name!r} is used twice")
            backend = entry.get("backend", MODEL_SECTIONS[section])
            _require(backend in (*BACKENDS, "custom"), f"model.{section}.{name}: unknown backend {backend!r}")
            models[str(name)] = {**entry, "backend": backend}
    return models


def _parse_ner(raw, models: dict[str, dict[str, Any]]) -> NerConfig | None:
    if not raw:
        return None
    _require(isinstance(raw, dict), "ner must be a mapping")
    settings = dict(raw)
    override = dict(settings.pop("params", None) or {})
    label_map = dict(settings.pop("label_map", None) or {})

    if "class" in settings:   # inline custom recognizer: ner: {class: pkg.mod:Cls, params: {...}}
        return NerConfig(class_path=str(settings["class"]), params=override, label_map=label_map,
                         min_score=float(settings.get("threshold", settings.get("min_score", 0.0))))

    name = settings.pop("model", None)
    _require(name is not None, "ner needs a model (a name under model.*) or a class")
    _require(name in models, f"ner.model {name!r} is not defined under model.*; available: {sorted(models)}")
    entry = dict(models[name])
    backend = entry.pop("backend")

    if backend == "custom":
        _require("class" in entry, f"model.custom.{name} needs a class")
        return NerConfig(class_path=str(entry["class"]), params={**(entry.get("params") or {}), **override},
                         label_map=label_map, model=name, backend=backend,
                         min_score=float(settings.get("threshold", 0.0)))

    required = "model_dir" if backend == "hf_bio" else "model"
    _require(required in entry, f"model {name!r} ({backend}) needs {required}")
    rename = NER_SETTING_NAMES[backend]
    params = {rename.get(k, k): v for k, v in entry.items()}
    params.update({rename.get(k, k): v for k, v in settings.items()})
    params.update(override)
    if label_map:
        params["label_map"] = label_map
    return NerConfig(class_path=BACKENDS[backend], params=params, model=name, backend=backend)


def parse_config(raw: dict[str, Any] | None, base_dir: str | Path | None = None) -> DetectorConfig:
    raw = raw or {}
    _require(isinstance(raw, dict), "the config must be a mapping")
    for key in raw:
        if key in MOVED_KEYS:
            raise ConfigError(f"'{key}' has moved to {MOVED_KEYS[key]}")
        _require(key in TOP_LEVEL_KEYS, f"unknown key {key!r}; known: {sorted(TOP_LEVEL_KEYS)}")

    languages = raw.get("languages") or ["en"]
    _require(isinstance(languages, list) and all(isinstance(x, str) for x in languages),
             "languages must be a list of strings")
    default_language = raw.get("default_language", languages[0])
    _require(default_language in languages, "default_language must be one of languages")

    base = Path(base_dir) if base_dir is not None else None
    patterns = []
    if raw.get("regex_patterns_file"):
        patterns += _load_pattern_file(str(raw["regex_patterns_file"]), base)
    patterns += _parse_patterns(raw.get("regex_patterns"), "regex_patterns")

    models = _parse_models(raw.get("model"))
    return DetectorConfig(
        languages=languages,
        default_language=default_language,
        presidio_recognizers=_parse_presidio(raw.get("presidio_recognizers")),
        patterns=patterns,
        anonymizer=_parse_anonymizer(raw.get("anonymizer")),
        ner=_parse_ner(raw.get("ner"), models),
        models=models,
    )


def default_config_path() -> Path:
    """The detector_config.yaml shipped inside the package."""
    return Path(str(resources.files("pii_pipeline.config").joinpath("detector_config.yaml")))


def load_config(path: str | Path | None = None) -> DetectorConfig:
    path = Path(path) if path else default_config_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file {str(path)!r} not found") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    return parse_config(raw, base_dir=path.parent)
