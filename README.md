# pii-pipeline

PII detection and masking built on [Microsoft Presidio](https://microsoft.github.io/presidio/).
Every detector is a Presidio recognizer: a NER model (BIO token classification or GLiNER), regex
patterns, deny lists, Presidio's own recognizers, and labels taken from document structure (tables,
`Label: value` lines). Span rules and context checks clean the detections up, and the anonymizer
replaces them with indexed placeholders:

```
Hi Tan Wei Ming (陈伟明), account 123-456789-001  ->  Hi [PERSON_01] ([PERSON_02]), account [ACCOUNT_01]
```

The pipeline is model-agnostic: the NER model is chosen by name in the config. The HTTP API is
documented in [docs/API.md](docs/API.md).

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Masking modes](#masking-modes)
- [Structured labels: tables and inline labels](#structured-labels-tables-and-inline-labels)
- [Repeated values and name variants](#repeated-values-and-name-variants)
- [Span rules](#span-rules)
- [NER models: BIO and GLiNER](#ner-models-bio-and-gliner)
- [Python API](#python-api)
- [Command line](#command-line)
- [HTTP API](#http-api)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)

## Quick start

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # GPU build
.venv/Scripts/python -m pip install -e ".[dev]"                                                       # Windows
.venv/Scripts/python -m pip install -e ".[dev,gliner]"                                                # + GLiNER models
# Linux / macOS: .venv/bin/python instead of .venv/Scripts/python
```

Torch is pinned to 2.8, the version the v5.0 model was trained with. Install it before the editable
install if you want a specific build. CPU works too, with `ner.device: cpu` in the config.

### 2. Get the NER model

The model is not in this repo. Extract the v5.0 release into `models/`:

```bash
unzip ../token_classification_ft/outputs/release/xlmr-base-pii-ml-v5.0.zip -d models/xlmr-base-pii-ml-v5.0
```

### 3. Run it

```bash
python -m pii_pipeline.cli --text "Hi Tan Wei Ming, account 123-456789-001"
uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

Both use the config shipped in the package (`src/pii_pipeline/config/detector_config.yaml`) unless
you pass `--config` / set `PII_CONFIG`.

## How it works

**Detection** (`PIIPipeline.detect`):

1. **Detectors.** Presidio's `AnalyzerEngine` runs every recognizer on the no-op NLP engine (no spaCy
   model needed): deny lists, table and inline labels, regex patterns, Presidio built-ins, and the
   NER model. Each is registered once per language; the copies share the loaded model. The allow
   list is applied here.
2. **Span rules.** Each detection must pass the rules for its label and the `"*"` rules. Deny-list
   values are not checked. Refused detections are reported in `rejected`.
3. **Context check.** A NER, regex or Presidio match that cuts an ASCII word is grown to the whole
   word when it covers more than half of it, since that is usually a boundary error ("Tan Wei Min" in
   "Tan Wei Ming", "12345678" in "acct12345678"). A smaller fragment ("Ann" in "Annual") is rejected
   as `context:inside_word`. For `separator_insensitive` labels, a span that stops at `-` or `.`
   inside a longer value ("123-456" in "123-456-7") is grown over the rest of it.
4. **Structure overrides models.** Where a table cell or inline-label value overlaps a model span,
   the structured label wins on the text it covers (`masking/semantic_values.py`).
5. **Overlap merging.** Overlapping spans merge into their union. The label comes from the best
   member by `priority`, then score, then length. Nothing detected is left visible.

**Masking** (`PIIAnonymizer`):

6. Spans whose values match under the label's variant rules share an identity.
7. Other copies of values in `entities_to_look_out_for` are masked too; with `optimized: true`,
   short forms of names as well.
8. Neighbouring spans of one identity separated only by spaces merge into one.
9. Tokens are assigned per identity (or per exact surface in versioned mode) and the text is built.
   Labels with a `per_entity` operator are passed to Presidio's `AnonymizerEngine` instead.

## Configuration

Two files in `src/pii_pipeline/config/`:

- **`detector_config.yaml`**: everything except regexes.
- **`regex_patterns.yaml`**: your regex detectors, `{label, name, pattern, score?, context?}`. Ships
  with commented examples only (email, SG NRIC, phone). It is found relative to the config file.

`detector_config.yaml` sections:

| Section / key | Purpose |
|---|---|
| `languages`, `default_language` | Languages accepted per request. |
| `regex_patterns_file` | Regex file, relative to the config. `regex_patterns:` can also list patterns inline. |
| `presidio_recognizers.builtin` | Presidio predefined recognizers by class name, e.g. `[EmailRecognizer, IbanRecognizer]`. |
| `presidio_recognizers.deny_list` | Exact values to mask, score 1.0. Case-insensitive unless `case_sensitive: true`. |
| `anonymizer.mode`, `format`, `per_entity` | Placeholder mode and template; Presidio operators per label. See [Masking modes](#masking-modes). |
| `anonymizer.priority` | Source order for labelling merged spans. Unlisted sources rank last. |
| `anonymizer.allow_listed_values` | Optional. Exact strings that are never masked. |
| `anonymizer.entities_to_look_out_for` | Labels whose values are masked wherever they repeat. |
| `anonymizer.separator_insensitive` | Labels whose values ignore separators (`123-456` = `123456`). |
| `anonymizer.span_variant_rules` | Per-label variant rules: `case`, `whitespace`, `separators`, `linebreaks`. |
| `anonymizer.unmask_url` | `true` leaves spans that sit inside a URL unmasked. |
| `anonymizer.included_entities` | Mask only these labels (`null` = all). |
| `anonymizer.optimized` | Also mask short forms of detected names. |
| `anonymizer.span_alias` | Inline labels: `{LABEL: [alias, ...]}` for `Alias: value` lines. |
| `anonymizer.table_semantics` | Table columns and rows: `{name, entity_type, aliases}`. |
| `anonymizer.span_rules` | Per-label rules a span must pass; `"*"` for all labels. |
| `ner` | The model in use (`model: <name>`) and settings that override its entry: `device`, `window_size`, `stride`, `threshold`, `label_map`. |
| `model` | Every available model: `token_classification`, `gliner`, `custom`. |

Old top-level keys (`allow_list`, `masking`, `deny_list`, `patterns`, `tables`, `checks`, ...) are
rejected with a message naming their new place.

## Masking modes

Both modes give the same index to variants of one value (case, spacing, and the label's other
variant rules). They differ in how placeholders map back to the text.

| | `original` (default) | `versioned` |
|---|---|---|
| Placeholder | one per value: `[PERSON_01]` | one per exact text: `[PERSON_01_1]`, `[PERSON_01_2]` |
| `Tan Wei Ming` and `tan wei ming` | both `[PERSON_01]` | `[PERSON_01_1]` and `[PERSON_01_2]` |
| Mapping back | lossy: keeps the first variant only | 1:1: each placeholder maps to exactly one original |

Default formats are `[{entity}_{index:02d}]` and `[{entity}_{index:02d}_{version}]`. A custom
`format` must contain `{entity}` and `{index}`, plus `{version}` in versioned mode.
`masking/tokens.py` builds, finds and parses tokens (`parse_token("[PERSON_01_2]") -> ("PERSON", 1, 2)`).
Use `unmask(masked, mapping)` to restore text. Mappings contain PII; store them as you would the source.

## Structured labels: tables and inline labels

Document structure says what a value is, which is more reliable than guessing from the value.
Structured labels override model labels on the text they cover.

**Tables** (`table_semantics`):

```yaml
table_semantics:
  columns:   # header-row cell -> labels every non-empty cell below it
    - {name: account, entity_type: ACCOUNT, aliases: [account no, acct]}
  rows:      # first cell of a row -> labels the other non-empty cells in that row
    - {name: client, entity_type: PERSON, aliases: [client name]}
```

- `| Account No | Note |`: the cells under "Account No" are ACCOUNT.
- `| Client name | Tan Wei Ming |`: the cells to the right of "Client name" are PERSON.
- Header and row-header cells are never masked. A cell is labelled as a whole.
- A model span that crosses into a labelled cell is cut at the cell, so it never joins two cells.

**Inline labels** (`span_alias`), outside tables:

```
Client name: Tan Wei Ming          colon
- DOB: 1990-01-01                  bullet (-, *, +, •, 1.)
**Phone:** +65 6123 4567           emphasis is ignored
Name: Tan Wei Ming; DOB: 1990-01-01   several pairs on one line
```

A value runs to the end of the line, a `;` or `|`, or the next known label. Aliases match exactly
after normalising case, spacing and `*` `_` `` ` `` `:` emphasis; substrings don't match.

## Repeated values and name variants

**Repeated values.** For labels in `entities_to_look_out_for`, once a value is detected every other
copy in the document is masked with the same identity, in any format its variant rules accept:
`TAN WEI MING` for `Tan Wei Ming`, `123 456` for `123-456` (separator-insensitive labels), and
e-mail addresses split across lines (`john.doe@exam⏎ple.com`). Copies inside a longer word or a
longer separated value (`123-456` in `123-456-7`) and allow-listed values are skipped. A single
capitalised word (`Will`, `May`) is matched with its exact case, so the ordinary lowercase word is
left alone. These spans have source `propagated`.

**Name variants** (`optimized: true`). For each detected name (`name_entities`, default `[PERSON]`)
the anonymizer also masks `Wei Ming`, `Tan`, `Mr Tan`, `W. M. Tan` and similar forms, with the full
name's identity. To avoid tagging a different person:

- a form shared by two detected names is skipped (`Tan` when both `Tan Wei Ming` and `Tan Ah Kow` are found)
- a form next to a capitalised word that is not part of the name is skipped (`Tan` in `Tan Ah Kow`);
  titles and sentence-initial words don't count
- forms are matched case-sensitively, as whole words, longest first

These spans have source `name_variant`.

## Span rules

A span rule is a check a detection must pass before it is masked: `fn(text, params, today) -> bool`,
`True` keeps the span.

```yaml
span_rules:
  "*":  [{rule: min_length, chars: 2}]           # every label
  DOB:  [{rule: dob_adult, years: 18}]            # built-in rule
  TIN:  [{rule: "myrules.ids:valid_nric"}]        # your own function
```

Built-in rules (`masking/span_rules.py`): `min_length`, `max_length`, `has_digit`, `has_letter`,
`matches` (regex), `not_in` (values), `dob_adult`, `not_future`. Date rules read ISO and day-first
dates and keep values they cannot parse.

- A span is masked only if every rule for its label and every `"*"` rule returns `True`.
- Rules are resolved when the config loads, so a bad name fails there. Built-in rules are also run
  once on sample values, so bad params (a missing `pattern`, `chars: two`) fail there too.
- A rule that raises during a request counts as passed: the span is still masked, and a warning
  (without the text) is logged.
- A rejected detection is reported in `rejected` with its rule name and no text.
- Deny-list values are not checked. Rules run before merging, so a rejected span never masks text.

## NER models: BIO and GLiNER

Models are listed under `model` and chosen with `ner.model`:

```yaml
ner:
  model: gliner_multi_pii      # switch models by name
  device: auto
model:
  token_classification:        # Hugging Face BIO token classification
    xlmr_v5: {model_dir: models/xlmr-base-pii-ml-v5.0, window_size: 384, stride: 96}
  gliner:                      # GLiNER, needs `pip install .[gliner]`
    gliner_multi_pii:
      model: urchade/gliner_multi_pii-v1
      threshold: 0.5
      labels: {person: PERSON, "phone number": PHONE}
  custom:                      # any Presidio EntityRecognizer
    mine: {class: "my_pkg.rec:MyRecognizer", params: {}}
```

- **BIO** (`recognizers/ner_hf.py`): sliding windows (`window_size` in tokens), max-confidence merge,
  visible line breaks, script word boundaries and punctuation trim, as in the v5.0 release.
  `tests/test_ner_v5.py` checks parity with the sibling `PIIPredictor`.
- **GLiNER** (`recognizers/gliner.py`): `labels` maps GLiNER's prompt labels to entity types (a
  plain list uses the upper-cased label). Long text is cut into overlapping character windows
  (`window_size` in characters, `stride`), preferably at whitespace. If the `gliner` package is
  missing, loading fails with a message saying how to install it.
- **Custom**: a Presidio `EntityRecognizer` whose constructor takes `params`. An inline
  `ner: {class: ..., params: ...}` also works without a `model` entry.

`ner` settings override the same keys in the chosen entry, so `threshold` or `window_size` set there
apply to whichever model is in use.

## Python API

```python
from pii_pipeline import PIIPipeline, DetectionRequest

pipeline = PIIPipeline.from_yaml()                # packaged config, or a path
found = pipeline.detect(DetectionRequest(text, context={"language": "en"}))
result = pipeline.mask(text, return_mapping=True)  # result.text, result.entities, result.mapping
```

`detect` returns a `DetectionResult` (text, sha256 checksum, `PIISpan`s, `rejected`). To keep
tokens consistent across documents, reuse one anonymizer:

```python
anon = pipeline.new_anonymizer()
anon.load_mappings({"Tan Wei Ming": "[PERSON_07]"})   # optional: user-chosen tokens
for doc in docs:
    masked = pipeline.mask(doc, anonymizer=anon).text
anon.mappings()            # surface -> token for everything seen
anon.mappings(doc)         # only surfaces that occur in doc
```

`anon.resolve(result, text)` assigns tokens without building text; `anonymize` refuses a result
whose checksum doesn't match the text. Errors are `PIIError` subclasses in `core/errors.py`, each
with a `code` and HTTP status.

## Command line

```bash
python -m pii_pipeline.cli --text "..."
python -m pii_pipeline.cli --config my_config.yaml --file doc.txt --json
python -m pii_pipeline.cli --file doc.txt --json --mapping   # mapping contains PII
```

`--json` prints `text`, `entities`, and `rejected`. `--mapping` adds `mapping`.

## HTTP API

```bash
uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, languages, NER model, span-rule count, masking mode |
| `POST` | `/analyze` | Detect spans. Nothing is masked. |
| `POST` | `/mask` | Detect and mask. Optional `return_mapping`. |

Full reference, with request and response examples, error codes, and offset rules, is in
[docs/API.md](docs/API.md). Interactive docs are served at `/docs`.

The API has no authentication. Run it on a private network or behind a gateway.

### Docker

The image runs the API on CPU with the packaged config and does not contain the model weights.
Mount `models/` at runtime:

```bash
docker build -t pii-mask-pipeline .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" pii-mask-pipeline
```

On Windows with Git Bash, prefix the run command with `MSYS_NO_PATHCONV=1`. Otherwise Git Bash
rewrites the container path in `-v` and the mount is missing.

The container runs as a non-root user and reports health at `/health`. It uses one worker, because
each worker loads its own copy of the model. To use another config, mount it and pass
`-e PII_CONFIG=/path/in/container/detector_config.yaml`.

## Tests

```bash
.venv/Scripts/python -m pytest                        # everything
.venv/Scripts/python -m pytest tests/test_anonymizer.py   # no model needed
```

Model-based tests skip when `models/` is missing. Most of the suite uses a fake NER
(`tests/fake_ner.py`) or a fake `gliner` module and runs in seconds.

## Known limitations

- **Merging can over-mask.** A NER span that overlaps a span of another type takes that span's label,
  and the union is masked as one placeholder. Structured labels are the exception: they cut model
  spans instead of merging with them.
- **Inline values run to the end of the line** unless a `;`, `|` or another known label ends them.
  Keep `span_alias` to labels that introduce a single value.
  Long values are masked whole. A value on the line after its label is not found.
- **Name variants are for space-separated names.** Names written without spaces (Chinese, Japanese)
  get no short forms, and lowercase particles (`bin`, `de`) are never forms on their own. The
  neighbour guard is a heuristic: a capitalised word after a short form blocks it, and a short form
  that starts a sentence is masked even when it is an ordinary word ("Mark the date" for Mark Tan).
- **Repeated-value masking is document-wide.** A detected value in `entities_to_look_out_for` is
  masked at every whole-word occurrence. A single-word name that is also a common word is still
  masked at the start of a sentence ("May I help?" when "May" is a detected name).
- **Exact table aliases.** A header like `Account No (SGD)` doesn't match the alias `account no`, and
  placeholder cells such as `N/A` under a labelled column are masked too.
- **Case-insensitive placeholders in original mode.** `Jane Doe` and `jane doe` share `[PERSON_01]`,
  and the mapping keeps the first original seen. Versioned mode keeps them apart.
- **Masking mode is set in config only.** Requests cannot choose a mode per call.
- **Single-process server.** One document per request, no batching, no authentication or rate
  limiting.

## Project layout

```
docs/API.md                    HTTP API reference
models/                        NER weights (not in repo)
src/pii_pipeline/
  api.py                       FastAPI app: /health, /analyze, /mask
  cli.py                       command line
  pipeline.py                  PIIPipeline: detect, mask
  analyzer.py                  Presidio analyzer assembly, NER loading, merge_overlaps
  registry.py                  'module:Class' loader
  config/
    detector_config.py         YAML -> validated DetectorConfig
    detector_config.yaml       default configuration
    regex_patterns.yaml        regex detectors (examples)
  core/
    types.py                   DetectionRequest, PIISpan, DetectionResult
    errors.py                  PIIError and subclasses, with codes and HTTP statuses
  masking/
    anonymizer.py              PIIAnonymizer: identities, tokens, propagation, unmask
    optimized_anonymizer.py    OptimizedAnonymizer: name variants with neighbour guard
    tokens.py                  placeholder format, parse, validate
    span_variants.py           variant rules: case, whitespace, separators, line breaks
    span_rules.py              built-in rules and the rule gate
    structured_rules.py        markdown tables and inline 'Label: value' parsing
    semantic_values.py         structured labels override model spans
  nlp/
    context_enhancer.py        word-fragment check, label-in-window checks
  recognizers/
    ner_hf.py, _bio.py         BIO token-classification recognizer (v5.0 port)
    gliner.py                  GLiNER adapter
    deterministic.py           deny-list, regex (with context words), Presidio built-ins
    table.py                   structured-label recognizer (tables, inline labels)
tests/                         pytest suite
```
