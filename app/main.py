"""FastAPI shell — a thin web layer over the rag_core package.

Run locally:  uv run uvicorn app.main:app --reload
Swagger UI:   http://localhost:8000/docs
Dashboard:    http://localhost:8000/
"""

from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.deps import get_pipeline, require_api_key, save_and_ingest
from app.ui import router as ui_router
from rag_core.config import get_settings
from rag_core.llm import LLMNotConfiguredError

app = FastAPI(
    title="RAG Microservice",
    version="0.1.0",
    description=(
        "Ingest documents in any format (PDF, Word, PowerPoint, images with OCR) "
        "and ask questions answered with citations back to the source.\n\n"
        "Pipeline: Docling parsing → structure-aware chunking → hybrid search "
        "(dense + BM25) in Qdrant → cross-encoder reranking → Claude with citations.\n\n"
        "Authenticate with the `X-API-Key` header (Authorize button above)."
    ),
    openapi_tags=[
        {"name": "ops", "description": "Health and operational endpoints"},
        {"name": "documents", "description": "Upload and manage documents"},
        {"name": "query", "description": "Ask questions, get cited answers"},
    ],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.include_router(ui_router)


@app.middleware("http")
async def add_instance_header(request: Request, call_next):
    # Identifies which replica served the request — makes nginx/K8s
    # load balancing visible from the client side.
    response = await call_next(request)
    response.headers["X-Instance"] = get_settings().instance_name
    return response


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check used by load balancers and Kubernetes probes."""
    settings = get_settings()
    return {
        "status": "ok",
        "instance": settings.instance_name,
        "vector_store": "embedded" if not settings.qdrant_url else settings.qdrant_url,
    }


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@app.post(
    "/documents",
    tags=["documents"],
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def upload_document(file: UploadFile, pipeline=Depends(get_pipeline)) -> dict:
    """Upload a document (PDF, Word, PowerPoint, image, ...) and index it.

    Re-uploading a file with the same name replaces its previous version.
    """
    result = save_and_ingest(file, pipeline)
    return {
        "source": result.source,
        "chunks_indexed": result.num_chunks,
        "pages": result.num_pages,
    }


@app.get("/documents", tags=["documents"], dependencies=[Depends(require_api_key)])
def list_documents(pipeline=Depends(get_pipeline)) -> dict:
    """List the indexed documents."""
    return {"documents": pipeline.list_sources()}


@app.delete(
    "/documents/{source}",
    tags=["documents"],
    status_code=204,
    dependencies=[Depends(require_api_key)],
)
def delete_document(source: str, pipeline=Depends(get_pipeline)) -> None:
    """Remove a document and all its chunks from the index."""
    pipeline.delete_source(source)
    stored = Path(get_settings().uploads_dir) / Path(source).name
    stored.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["What is the refund policy?"])


class CitationOut(BaseModel):
    marker: int
    source: str
    page: int | None
    heading: str | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    model: str
    citations: list[CitationOut]


@app.post(
    "/query",
    tags=["query"],
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
)
def query(body: QueryRequest, pipeline=Depends(get_pipeline)) -> QueryResponse:
    """Ask a question; the answer cites the source passages it used."""
    answer = pipeline.query(body.question)
    return QueryResponse(
        answer=answer.text,
        model=answer.model,
        citations=[
            CitationOut(
                marker=c.marker,
                source=c.chunk.source,
                page=c.chunk.page,
                heading=c.chunk.heading,
                text=c.chunk.text,
            )
            for c in answer.citations
        ],
    )


# --------------------------------------------------------------------------
# Error mapping: most-specific first — each upstream failure mode gets an
# honest HTTP status instead of a generic 500.
# --------------------------------------------------------------------------


@app.exception_handler(LLMNotConfiguredError)
async def handle_llm_not_configured(request: Request, exc: LLMNotConfiguredError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(anthropic.AuthenticationError)
async def handle_bad_key(request: Request, exc: anthropic.AuthenticationError):
    return JSONResponse(status_code=401, content={"detail": "Anthropic API key is invalid."})


@app.exception_handler(anthropic.RateLimitError)
async def handle_rate_limit(request: Request, exc: anthropic.RateLimitError):
    return JSONResponse(
        status_code=429, content={"detail": "Anthropic rate limit hit — retry shortly."}
    )


@app.exception_handler(anthropic.APIStatusError)
async def handle_api_status(request: Request, exc: anthropic.APIStatusError):
    return JSONResponse(
        status_code=502, content={"detail": f"Upstream model error ({exc.status_code})."}
    )


@app.exception_handler(anthropic.APIConnectionError)
async def handle_api_connection(request: Request, exc: anthropic.APIConnectionError):
    return JSONResponse(status_code=502, content={"detail": "Could not reach the Anthropic API."})
