# PII masking API. CPU image; the NER weights are mounted at runtime, not baked in.
#
#   docker build -t pii-mask-pipeline .
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" pii-mask-pipeline
#
# models/ must contain the v5.0 release (see README "Quick start"). The config's model_dir points
# at /app/models/xlmr-base-pii-ml-v5.0.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PII_CONFIG=/app/configs/pipeline.yaml

WORKDIR /app

# CPU-only torch, pinned to the version the model was trained with. The CUDA wheel from PyPI would add
# several GB. pyproject's unpinned "torch" is then already satisfied, so pip does not pull the CUDA build.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install .

# Run as an unprivileged user. models/ is created here so a mount has a target.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/models \
 && chown -R app:app /app
USER app

EXPOSE 8000

# The first start loads the model, so allow a long start period before failing the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker: each worker process loads its own copy of the NER model.
CMD ["uvicorn", "pii_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
