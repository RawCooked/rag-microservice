"""Shared FastAPI dependencies: the pipeline singleton, API-key auth,
and the upload helper used by both the JSON API and the dashboard."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from rag_core.config import get_settings
from rag_core.parsing import SUPPORTED_EXTENSIONS
from rag_core.pipeline import IngestResult

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    # Built lazily on first use: loads the parsing + embedding models.
    # Tests replace this via app.dependency_overrides — no models needed.
    #
    # Double-checked locking, not lru_cache: two concurrent first requests
    # would otherwise both build a pipeline, and the embedded Qdrant store
    # allows only one client per process (the second gets a lock error).
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from rag_core.pipeline import build_pipeline

                _pipeline = build_pipeline(get_settings())
    return _pipeline


# Declared via Security() so Swagger UI shows an "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def save_and_ingest(file: UploadFile, pipeline) -> IngestResult:
    """Validate, persist, and index an uploaded file."""
    filename = Path(file.filename or "").name  # strip any client-sent path
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    uploads_dir = Path(get_settings().uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = uploads_dir / filename
    destination.write_bytes(file.file.read())
    return pipeline.ingest(destination)
