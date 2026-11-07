# PII masking API

HTTP API for detecting and masking PII in text. It runs the pipeline in [README.md](../README.md):
the NER model, deterministic rules, table headers, checks, and masking.

Interactive documentation is served by the running app at `/docs` (Swagger UI) and `/redoc`.
The OpenAPI schema is at `/openapi.json`.

## Running

```bash
uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

| Environment variable | Default | Meaning |
|---|---|---|
| `PII_CONFIG` | packaged `detector_config.yaml` | Detector configuration. Read once at startup. |
| `PII_MAX_TEXT_CHARS` | `200000` | Largest accepted `text`, in characters. Larger requests get `413`. |

The model loads once, when the server starts. A bad config or a missing model stops startup,
so the server never answers with a half-loaded pipeline.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status and the active configuration summary |
| `POST` | `/analyze` | Detect PII spans. Nothing is masked. |
| `POST` | `/mask` | Detect and mask PII. Returns masked text. |

### `GET /health`

```json
{
  "status": "ok",
  "languages": ["en", "zh", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"],
  "default_language": "en",
  "ner": "xlmr_v5",
  "checks": 0,
  "masking_mode": "original"
}
```

| Field | Meaning |
|---|---|
| `status` | Always `ok` while the server is up. |
| `languages` | Languages accepted in `language`. |
| `default_language` | Used when a request omits `language`. |
| `ner` | Name of the configured NER model (`ner.model`), its class path for an inline `ner.class`, or `null` if none. |
| `checks` | Number of configured span rules (`anonymizer.span_rules`). |
| `masking_mode` | `original` or `versioned`. See [Masking modes](#masking-modes). |

### `POST /analyze`

Detection only. Use it to review what would be masked, or to build a review step in front of
masking. The response has no matched text and no placeholders.

Request:

```json
{
  "text": "Tan Wei Ming, account 123-456789-001, call +65 6123 4567.",
  "language": "en"
}
```

Response (`200`):

```json
{
  "entities": [
    {"entity_type": "PERSON",  "start": 0,  "end": 12, "score": 0.9998, "source": "ner"},
    {"entity_type": "ACCOUNT", "start": 22, "end": 36, "score": 0.9999, "source": "ner"},
    {"entity_type": "PHONE",   "start": 43, "end": 56, "score": 1.0,    "source": "ner"}
  ],
  "rejected": []
}
```

Use `start` and `end` with the text you sent to recover each match, for example
`text[start:end]` in Python. See [Offsets](#offsets).

### `POST /mask`

Detect and mask. Request:

```json
{
  "text": "Tan Wei Ming, account 123-456789-001, call +65 6123 4567.",
  "language": "en",
  "return_mapping": true
}
```

Response (`200`):

```json
{
  "text": "[PERSON_01], account [ACCOUNT_01], call [PHONE_01].",
  "entities": [
    {"entity_type": "PERSON",  "start": 0,  "end": 12, "score": 0.9998, "source": "ner", "placeholder": "[PERSON_01]"},
    {"entity_type": "ACCOUNT", "start": 22, "end": 36, "score": 0.9999, "source": "ner", "placeholder": "[ACCOUNT_01]"},
    {"entity_type": "PHONE",   "start": 43, "end": 56, "score": 1.0,    "source": "ner", "placeholder": "[PHONE_01]"}
  ],
  "rejected": [],
  "mapping": {
    "[PERSON_01]": "Tan Wei Ming",
    "[ACCOUNT_01]": "123-456789-001",
    "[PHONE_01]": "+65 6123 4567"
  }
}
```

`mapping` is `null` unless `return_mapping` is `true`. **The mapping contains the original PII.**
Request it only when you need to restore the text, and store it with the same care as the source.

## Request fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string | yes | | Document to process. Up to `PII_MAX_TEXT_CHARS` characters. Empty text is allowed. |
| `language` | string or null | no | `default_language` | Must be one of the configured `languages`. |
| `return_mapping` | boolean | no | `false` | `/mask` only. Adds `mapping`, which contains PII. |

Unknown fields are ignored.

## Response fields

### Entity

| Field | Type | Meaning |
|---|---|---|
| `entity_type` | string | Label, e.g. `PERSON`, `ACCOUNT`, `DOB`. After any `label_map` renaming. |
| `start`, `end` | integer | Character offsets of the span. `end` is exclusive. |
| `score` | number | Confidence, rounded to 4 decimal places. Deny-list matches are `1.0`. |
| `source` | string | Which detector produced the span: `deny_list`, `table_columns`, `table_rows`, `inline_labels`, `patterns`, `presidio`, or `ner`. Spans the anonymizer added are `propagated` (another copy of a detected value) or `name_variant` (a short form of a detected name). |
| `placeholder` | string | `/mask` only. The replacement text. |

Entities are sorted by `start`. Overlapping detections are merged into one entity, so no two
entities overlap. `/mask` can list more entities than `/analyze` for the same text: it also
masks other copies of values in `entities_to_look_out_for` and, with `optimized: true`, short forms
of names. `/analyze` lists only what the detectors found.

### Rejected

A detection that a span rule or the context check refused. It was not masked and does not appear in `entities`.

| Field | Type | Meaning |
|---|---|---|
| `entity_type` | string | Label the detection had. |
| `start`, `end` | integer | Offsets of the refused span. |
| `source` | string | Recognizer that produced it. |
| `rule` | string | The rule that refused it: a built-in name, a `package.module:function` path, or `context:inside_word` for a small fragment of a longer word. |

Rejected entries contain no matched text. Deny-list values are never rejected.

## Masking modes

Set by `anonymizer.mode` in the configuration. The API does not choose a mode per request.

| Mode | Placeholder | Mapping back |
|---|---|---|
| `original` (default) | one per value: `[PERSON_01]` | Lossy. Variants such as `Tan Wei Ming` and `tan wei ming` share one placeholder, and the mapping keeps only the first. |
| `versioned` | one per exact text: `[PERSON_01_1]`, `[PERSON_01_2]` | 1:1. Every placeholder maps to exactly one original string. |

In versioned mode, a response from `/mask` with `return_mapping: true` can be reversed exactly
by replacing each placeholder with its mapping value.

## Errors

Pipeline errors return `{"detail": "<message>", "code": "<code>"}`. Request validation errors use
FastAPI's standard `{"detail": [...]}` body.

| Status | `code` | When | Example `detail` |
|---|---|---|---|
| `413` | `text_too_large` | `text` is longer than `PII_MAX_TEXT_CHARS` | `"text is 250000 characters; limit is 200000"` |
| `422` | `invalid_text` | `text` contains a lone UTF-16 surrogate (a JSON `\ud800`-style escape) | `"text has an invalid character (lone surrogate) at offset 12"` |
| `422` | `unsupported_language` | `language` is not configured | `"language 'fr' is not configured; available: ['en', ...]"` |
| `422` | | Request body is malformed or a required field is missing | Pydantic's validation list |
| `500` | | Unexpected server error | The request text is not included in logs or responses |

## Offsets

`start` and `end` are indexes into the Python `str` you sent: one unit per Unicode code point.
They are not byte offsets and not UTF-16 units. For Python, `text[start:end]` recovers the span.
JavaScript strings index by UTF-16 code unit, so characters outside the Basic Multilingual Plane
(some CJK extensions, emoji) shift offsets. Convert offsets on the client if that matters.

## Examples

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/mask \
  -H "Content-Type: application/json" \
  -d '{"text": "Tan Wei Ming, account 123-456789-001", "return_mapping": false}'
```

### Python

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:8000/mask",
    json={"text": "Tan Wei Ming, account 123-456789-001"},
    timeout=60,
)
r.raise_for_status()
masked = r.json()["text"]
```

### Restoring text (versioned mode)

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:8000/mask",
    json={"text": document, "return_mapping": True},
    timeout=60,
).json()

# store r["mapping"] securely; then:
restored = r["text"]
for placeholder, original in sorted(r["mapping"].items(), key=lambda kv: -len(kv[0])):
    restored = restored.replace(placeholder, original)
assert restored == document
```

Or use `pii_pipeline.unmask(r["text"], r["mapping"])` from the Python package, which does the same
in a single pass.

## Behaviour and limits

- **Synchronous.** One document per request. The response is returned when processing finishes.
  Requests run on a worker thread pool, so one slow request does not block `/health`.
- **No text in logs.** The service does not log request text or matched values.
- **No authentication or rate limiting.** Run it behind a gateway, or on a private network.
  Anyone who can reach the port can submit text.
- **Model runs on the server.** `/analyze` and `/mask` need the NER model loaded, so the first request
  after a cold start can be slower. `/health` answers without running the model.

## Versioning

The API is `version: 0.2.0`. 0.2.0 changed the versioned placeholder from `[PERSON_01.2]` to `[PERSON_01_2]` and added `code` to error bodies. Field additions are backward compatible. Removing or renaming a field
is a breaking change, and will bump the major version.
