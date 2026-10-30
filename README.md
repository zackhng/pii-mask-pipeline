# pii-pipeline

PII detection and masking built on [Microsoft Presidio](https://microsoft.github.io/presidio/).
A swappable NER model and deterministic rules are Presidio recognizers. Presidio's `AnalyzerEngine`
finds the spans, a checks gate and merge step clean them up, and `AnonymizerEngine` replaces them
with indexed placeholders:

```
Hi Tan Wei Ming (陈伟明), account 123-456789-001  ->  Hi [PERSON_01] ([PERSON_02]), account [ACCOUNT_01]
```

The pipeline is a template. The generic behaviour (merging, masking, the API) is the same for
every domain. What gets detected for a domain comes from a NER model plus the domain section of the
config. The HTTP API is documented in [docs/API.md](docs/API.md).

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Masking modes](#masking-modes)
- [Markdown tables](#markdown-tables)
- [Checks](#checks)
- [Command line](#command-line)
- [HTTP API](#http-api)
- [Swapping the NER model](#swapping-the-ner-model)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)

## Quick start

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # GPU build
.venv/Scripts/python -m pip install -e ".[dev]"                                                       # Windows
# Linux / macOS: .venv/bin/python instead of .venv/Scripts/python
```

Torch is pinned to 2.8, the version the model was trained with. Install it before the editable
install if you want a specific build. CPU works too, with `device: cpu` in the config.

### 2. Get the NER model

The model is not in this repo. Extract the v5.0 release into `models/`:

```bash
unzip ../token_classification_ft/outputs/release/xlmr-base-pii-ml-v5.0.zip -d models/xlmr-base-pii-ml-v5.0
```

### 3. Run it

```bash
python -m pii_pipeline.cli --config configs/pipeline.yaml --text "Hi Tan Wei Ming, account 123-456789-001"
PII_CONFIG=configs/pipeline.yaml uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

## How it works

1. **Recognizers.** Four kinds feed Presidio's analyzer:
   - the NER model (`HFTokenClassRecognizer`), the swap point
   - deny lists and regex patterns (stock `PatternRecognizer`s)
   - table recognizers, which label cells by their column or row headers
   
   Each recognizer is registered once per language. The copies share the loaded NER model.
2. **Analysis.** `AnalyzerEngine` runs on `NoOpNlpEngine`, so no spaCy model is needed.
   `allow_list` is applied by Presidio.
3. **Checks gate.** Each detection is checked against the rules for its label. A rejected detection
   is never masked and is reported in `rejected`. Deny-list values are not checked.
4. **Overlap merging.** Overlapping spans are merged into one span covering their union. The label
   comes from the best member, ordered by `priority`, then score, then length. Merging is
   transitive. Spans that only touch are not merged. Nothing detected is left visible.
5. **Masking.** Placeholders are assigned in text order, then Presidio's `AnonymizerEngine`
   applies them. Presidio runs operators in reverse text order, so indexes are assigned up front.
   The mode decides what counts as the same value. See [Masking modes](#masking-modes).

## Configuration

`configs/pipeline.yaml` has two parts, marked in the file:

- **TEMPLATE**: generic behaviour (`languages`, `default_language`, `allow_list`, `priority`,
  `masking`). The same for every domain.
- **DOMAIN**: what to detect (`ner`, `separator_insensitive`, `deny_list`, `tables`, `patterns`,
  `checks`). To adapt the pipeline to another domain, replace this part.

| Key | Purpose |
|---|---|
| `ner.class` | `package.module:ClassName` of the NER recognizer. This is the swap point. |
| `ner.params` | Passed to that class's constructor. For v5.0: `model_dir`, `device`, `max_length`, `stride`. |
| `ner.label_map` | Renames model labels, e.g. `{BUSINESS: ORGANIZATION}`. Forwarded only when set. |
| `ner.min_score` | Drops NER spans below this score. Forwarded only when set. |
| `separator_insensitive` | Labels whose values match across separators (`123-456` = `123456`). Affects placeholder indexes only. |
| `deny_list` | Exact values to mask, score 1.0. Case-insensitive unless `case_sensitive: true`. |
| `tables` | Column and row header aliases, inline or via `fields_file`. See [Markdown tables](#markdown-tables). |
| `patterns` | Regexes with a score. |
| `checks` | Per-label rules a detection must pass to be masked. See [Checks](#checks). |
| `allow_list` | Exact strings that are never masked. |
| `priority` | Source order for choosing the label of a merged span: source, then score, then length. Must list `deny_list`, `table_columns`, `table_rows`, `patterns`, `ner`. |
| `masking.mode` | `original` (default) or `versioned`. See [Masking modes](#masking-modes). |
| `masking.format` | Placeholder template. See [Masking modes](#masking-modes). |
| `masking.per_entity` | Overrides an entity with a Presidio operator, e.g. `{DOB: {operator: redact}}`. |

A few constants still live in the NER port: `NUMERIC` in `recognizers/_bio.py` (labels that lose a
trailing full stop) is set for the v5.0 labels. A different model with different labels should
provide its own rule.

## Masking modes

Both modes give the same index to variants of one value. Variants differ in case and spacing, and
in separators for labels listed in `separator_insensitive`. The modes differ in how placeholders
map back to the text.

| | `original` (default) | `versioned` |
|---|---|---|
| Placeholder | one per value: `[PERSON_01]` | one per exact text: `[PERSON_01.1]`, `[PERSON_01.2]` |
| `Tan Wei Ming` and `tan wei ming` | both `[PERSON_01]` | `[PERSON_01.1]` and `[PERSON_01.2]` |
| Mapping back | lossy: keeps the first variant only | 1:1: each placeholder maps to exactly one original |

Formats:

- `original`: `[{entity}_{index:02d}]`
- `versioned`: `[{entity}_{index:02d}.{version}]`

Versioned mode requires `{version}` in `masking.format`. Use `unmask(masked, mapping)` to restore
the original text. Mappings contain PII, so store them as you would the source.

## Markdown tables

Tables are labelled from their headers, which is more reliable than guessing from the value.
Define the aliases in the `tables` section of `configs/pipeline.yaml`, in two sections:

```yaml
tables:
  column_headers:            # header-row cell -> labels every non-empty cell below it
    ACCOUNT: [account no, acct]
  row_headers:               # first cell of a row -> labels the other non-empty cells in that row
    PERSON: [client name]
```

Or keep them in a separate file with `tables.fields_file`. Use one or the other, not both.

- **Column mode:** in `| Account No | Note |`, the cells under "Account No" are ACCOUNT.
- **Row mode:** in `| Client name | Tan Wei Ming |`, the cells to the right of "Client name" are PERSON.
- Header and row-header cells are never masked.
- Aliases match exactly after normalising case, spacing, and `*`, `_`, `` ` ``, `:` emphasis. Substrings don't match.
- A cell is labelled as a whole. `Tax ID: S1234567D` in a labelled cell masks the whole cell.
- Table labels outrank NER on the same text. Their score is `tables.score`.

## Checks

A check is a rule a detection must pass before it is masked. It is a function you write, in your own
module. The template ships none. Enable them under `checks:`, one list per label:

```yaml
checks:
  DOB:
    - {rule: "myrules.dates:dob_adult", years: 18}
```

```python
# myrules/dates.py
from datetime import date

def dob_adult(text, params, today):
    """Return True to keep (mask) the span, False to reject it."""
    ...
```

- The rule gets the span text, its params from config, and the reference date.
- A span is masked only if every rule for its label returns `True`.
- Rules are checked when the config loads. A bad path fails there, not during masking.
- A rejected detection is reported in `rejected` with its rule name and no text.
- Deny-list values are not checked.
- Rules run before merging, so a rejected span never masks text.

Rules decide their own edge cases. A common choice is to return `True` when a value can't be
parsed, so that such values are still masked.

## Command line

```bash
python -m pii_pipeline.cli --config configs/pipeline.yaml --text "..."
python -m pii_pipeline.cli --config configs/pipeline.yaml --file doc.txt --json
python -m pii_pipeline.cli --config configs/pipeline.yaml --file doc.txt --json --mapping   # mapping contains PII
```

`--json` prints `text`, `entities`, and `rejected`. `--mapping` adds `mapping`.

## HTTP API

```bash
PII_CONFIG=configs/pipeline.yaml uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, languages, NER class, check count, masking mode |
| `POST` | `/analyze` | Detect spans. Nothing is masked. |
| `POST` | `/mask` | Detect and mask. Optional `return_mapping`. |

Full reference, with request and response examples, error codes, and offset rules, is in
[docs/API.md](docs/API.md). Interactive docs are served at `/docs`.

The API has no authentication. Run it on a private network or behind a gateway.

### Docker

The image runs the API on CPU and does not contain the model weights. Mount `models/` at runtime:

```bash
docker build -t pii-mask-pipeline .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" pii-mask-pipeline
```

The container runs as a non-root user and reports health at `/health`. It uses one worker, because
each worker loads its own copy of the model. Pass a different config with `-e PII_CONFIG=/app/configs/<file>`
after copying the file into the image, or by mounting it over `/app/configs`.

## Swapping the NER model

- **Different weights, same architecture:** change `ner.params.model_dir`.
- **Different implementation:** write a Presidio `EntityRecognizer` whose `analyze()` returns
  `RecognizerResult`s, then set `ner.class` to it. Its constructor receives `ner.params`.

The HF token-classification port (`recognizers/_bio.py`, `recognizers/ner_hf.py`) covers the
sliding windows, max-confidence merge, visible line breaks, script word boundaries, and punctuation
trim that the v5.0 release uses. The repo does not import the sibling project.
`tests/test_ner_v5.py` checks that spans match the sibling `PIIPredictor` on sample documents.

## Tests

```bash
.venv/Scripts/python -m pytest                               # everything
.venv/Scripts/python -m pytest tests/test_pipeline_rules.py  # no model needed
```

Model-based tests skip when `models/` is missing. Most of the suite uses a fake NER
(`tests/fake_ner.py`) and runs in seconds.

## Known limitations

- **Patterns have no context words.** With the no-op NLP engine, Presidio has no tokens to match
  context against, so context-dependent scoring is not available.
- **Merging can over-mask.** A NER span that overlaps a span of another type takes that span's label,
  and the union is masked as one placeholder. For example, a PERSON span touching an ADDRESS span
  becomes one ADDRESS placeholder. A deny-list value at the start of a longer span can also pull in
  trailing punctuation the NER span included.
- **Case-insensitive placeholders in original mode.** `Jane Doe` and `jane doe` share `[PERSON_01]`,
  and the mapping keeps the first original seen. Versioned mode keeps them apart.
- **Versioned placeholders need `{version}` in the format.** Configs that set a custom format without
  it fail to load.
- **Masking mode is set in config only.** Requests cannot choose a mode per call.
- **Single-process server.** One document per request, no batching, no authentication or rate
  limiting.

## Project layout

```
configs/pipeline.yaml        template + domain configuration
docs/API.md                  HTTP API reference
models/                      NER weights (not in repo)
src/pii_pipeline/
  api.py                     FastAPI app: /health, /analyze, /mask
  cli.py                     command line
  config.py                  YAML -> validated configuration
  analyzer.py                Presidio analyzer assembly, merge_overlaps
  pipeline.py                PIIPipeline: detect, mask
  masking.py                 placeholder assignment, modes, unmask
  checks.py                  user-rule loading and the gate
  tables.py                  markdown table parsing, header aliases
  registry.py                'module:Class' loader (swap point)
  recognizers/
    ner_hf.py                HF token-classification recognizer (v5.0 port)
    _bio.py                  BIO / window helpers for the NER port
    deterministic.py         deny-list and pattern recognizers
    table.py                 table-field recognizer
tests/                       pytest suite
```
