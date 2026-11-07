"""HTTP API: POST /analyze (detection only) and POST /mask (masked text).

Run:
    uvicorn pii_pipeline.api:app --host 127.0.0.1 --port 8000

Settings (environment variables, or arguments to create_app):
    PII_CONFIG          detector config YAML (default: the detector_config.yaml shipped in the package)
    PII_MAX_TEXT_CHARS  largest accepted text in characters (default 200000)

The API never logs request text. Entity results carry offsets, labels, scores and placeholders, not
the matched text. The placeholder -> original mapping is returned only when `return_mapping` is true,
and that mapping contains PII.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .core.errors import PIIError, TextTooLargeError, UnsupportedLanguageError
from .pipeline import PIIPipeline, entity_dict

DEFAULT_MAX_CHARS = 200_000


class TextIn(BaseModel):
    text: str = Field(..., description="Document text to analyse")
    language: str | None = Field(None, description="Language code; defaults to the config's default_language")


class MaskIn(TextIn):
    return_mapping: bool = Field(False, description="Include placeholder -> original mapping (contains PII)")


class EntityOut(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    source: str
    placeholder: str | None = None


class RejectedOut(BaseModel):
    entity_type: str
    start: int
    end: int
    source: str
    rule: str


class AnalyzeOut(BaseModel):
    entities: list[EntityOut]
    rejected: list[RejectedOut]


class MaskOut(BaseModel):
    text: str
    entities: list[EntityOut]
    rejected: list[RejectedOut]
    mapping: dict[str, str] | None = None


class HealthOut(BaseModel):
    status: str
    languages: list[str]
    default_language: str
    ner: str | None
    checks: int
    masking_mode: str


def create_app(config_path: str | None = None, max_text_chars: int | None = None,
               today=None) -> FastAPI:
    config_path = config_path or os.environ.get("PII_CONFIG") or None   # None -> packaged default
    max_chars = max_text_chars or int(os.environ.get("PII_MAX_TEXT_CHARS", DEFAULT_MAX_CHARS))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = PIIPipeline.from_yaml(config_path, today=today)
        yield

    app = FastAPI(title="PII masking API", version="0.2.0", lifespan=lifespan)

    @app.exception_handler(PIIError)
    async def pii_error(request: Request, exc: PIIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    def pipeline() -> PIIPipeline:
        return app.state.pipeline

    def check_request(body: TextIn) -> None:
        if len(body.text) > max_chars:
            raise TextTooLargeError(f"text is {len(body.text)} characters; limit is {max_chars}")
        if body.language is not None and body.language not in pipeline().config.languages:
            raise UnsupportedLanguageError(f"language {body.language!r} is not configured; "
                                           f"available: {pipeline().config.languages}")

    @app.get("/health", response_model=HealthOut)
    def health() -> dict[str, Any]:
        cfg = pipeline().config
        return {"status": "ok", "languages": cfg.languages, "default_language": cfg.default_language,
                "ner": cfg.ner.display_name if cfg.ner else None, "checks": len(cfg.anonymizer.span_rules),
                "masking_mode": cfg.anonymizer.mode}

    @app.post("/analyze", response_model=AnalyzeOut, response_model_exclude_none=True)
    def analyze(body: TextIn) -> dict[str, Any]:
        check_request(body)
        found = pipeline().detect(body.text, body.language)
        return {"entities": [entity_dict(s) for s in found.spans],
                "rejected": found.rejected}

    @app.post("/mask", response_model=MaskOut)
    def mask(body: MaskIn) -> dict[str, Any]:
        check_request(body)
        res = pipeline().mask(body.text, body.language, return_mapping=body.return_mapping)
        return {"text": res.text, "entities": res.entities, "rejected": res.rejected, "mapping": res.mapping}

    return app


app = create_app()
