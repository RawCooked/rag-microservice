# RAG Microservice

> Work in progress — being built phase by phase, see [docs/concepts](docs/concepts/) for the ideas behind each part.

A production-style Retrieval-Augmented Generation service:

- **Ingest anything**: PDF, Word, PowerPoint, images (OCR) — parsed layout-aware with Docling
- **State-of-the-art retrieval**: hybrid search (dense + BM25) in Qdrant, then cross-encoder reranking
- **Grounded answers**: Claude answers only from retrieved chunks, with citations back to document + page
- **Ops-ready**: Swagger UI, health checks, API-key auth, Docker + nginx load balancing, Kubernetes manifests
- **Verified**: unit tests (LLM mocked), CI, and a published retrieval eval (recall@k)

## Quickstart (dev)

```
uv sync --dev
uv run uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Configuration

Copy `.env.example` to `.env`. Every setting is an environment variable — see the file for docs.
