# PII masking API

HTTP API for detecting and masking PII in text. It runs the pipeline in [README.md](../README.md):
the NER model, deterministic rules, table headers, checks, and masking.

Interactive documentation is served by the running app at `/docs` (Swagger UI) and `/redoc`.
The OpenAPI schema is at `/openapi.json`.

## Running

```bash
PII_CONFIG=configs/pipeline.yaml uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000
```

| Environment variable | Default | Meaning |
|---|---|---|
| `PII_CONFIG` | `configs/pipeline.yaml` | Pipeline configuration. Read once at startup. |
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
  "ner": "pii_pipeline.recognizers.ner_hf:HFTokenClassRecognizer",
  "checks": 0,
  "masking_mode": "original"
}
```

| Field | Meaning |
|---|---|
| `status` | Always `ok` while the server is up. |
| `languages` | Languages accepted in `language`. |
| `default_language` | Used when a request omits `language`. |
| `ner` | Class path of the configured NER recognizer, or `null` if none. |
| `checks` | Number of configured check rules. |
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
| `source` | string | Which recognizer produced the span: `deny_list`, `table_columns`, `table_rows`, `patterns`, or `ner`. |
| `placeholder` | string | `/mask` only. The replacement text. |

Entities are sorted by `start`. Overlapping detections are merged into one entity, so no two
entities overlap.

### Rejected

A detection that a check rule refused. It was not masked and does not appear in `entities`.

| Field | Type | Meaning |
|---|---|---|
| `entity_type` | string | Label the detection had. |
| `start`, `end` | integer | Offsets of the refused span. |
| `source` | string | Recognizer that produced it. |
| `rule` | string | The `package.module:function` rule that refused it. |

Rejected entries contain no matched text. Deny-list values are never rejected.

## Masking modes

Set by `masking.mode` in the configuration. The API does not choose a mode per request.

| Mode | Placeholder | Mapping back |
|---|---|---|
| `original` (default) | one per value: `[PERSON_01]` | Lossy. Variants such as `Tan Wei Ming` and `tan wei ming` share one placeholder, and the mapping keeps only the first. |
| `versioned` | one per exact text: `[PERSON_01.1]`, `[PERSON_01.2]` | 1:1. Every placeholder maps to exactly one original string. |

In versioned mode, a response from `/mask` with `return_mapping: true` can be reversed exactly
by replacing each placeholder with its mapping value.

## Errors

Errors use FastAPI's standard body: `{"detail": ...}`.

| Status | When | Example `detail` |
|---|---|---|
| `413` | `text` is longer than `PII_MAX_TEXT_CHARS` | `"text is 250000 characters; limit is 200000"` |
| `422` | `language` is not configured | `"language 'fr' is not configured; available: ['en', ...]"` |
| `422` | Request body is malformed or a required field is missing | Pydantic's validation list |
| `500` | Unexpected server error | The request text is not included in logs or responses |

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

The API is `version: 0.1.0`. Field additions are backward compatible. Removing or renaming a field
is a breaking change, and will bump the major version.
