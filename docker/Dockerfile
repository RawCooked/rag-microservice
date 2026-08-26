# Multi-stage build: the builder stage installs dependencies with uv;
# the final stage carries only the virtualenv and the code — no build
# tools, smaller image, smaller attack surface.

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first: as long as pyproject/uv.lock don't change,
# Docker reuses this (large) cached layer on every rebuild.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY rag_core ./rag_core
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.11-slim

# opencv (pulled in by docling) needs these system libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd --create-home appuser \
    && mkdir -p /models /app/data \
    && chown -R appuser /models /app/data

COPY --from=builder --chown=appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    # Local ML models download here on first start; docker-compose mounts
    # a volume at /models so they survive container recreation.
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
