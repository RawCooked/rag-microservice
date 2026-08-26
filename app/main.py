"""FastAPI shell — a thin web layer over the rag_core package.

Run locally:  uv run uvicorn app.main:app --reload
Swagger UI:   http://localhost:8000/docs
"""

from fastapi import FastAPI, Request

from rag_core.config import get_settings

app = FastAPI(
    title="RAG Microservice",
    version="0.1.0",
    description=(
        "Ingest documents in any format (PDF, Word, PowerPoint, images with OCR) "
        "and ask questions answered with citations back to the source.\n\n"
        "Pipeline: Docling parsing → structure-aware chunking → hybrid search "
        "(dense + BM25) in Qdrant → cross-encoder reranking → Claude with citations."
    ),
    openapi_tags=[
        {"name": "ops", "description": "Health and operational endpoints"},
        {"name": "documents", "description": "Upload and manage documents"},
        {"name": "query", "description": "Ask questions, get cited answers"},
    ],
)


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
