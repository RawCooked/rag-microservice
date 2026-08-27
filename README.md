# RAG Microservice

![CI](https://github.com/RawCooked/rag-microservice/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style **Retrieval-Augmented Generation service**: upload documents in
any format, ask questions over them, and get answers **with citations back to the
exact source passage** — behind a clean REST API, a Swagger UI, and a zero-build
dashboard. Built without LangChain: every pipeline stage is ~100 lines of
readable, tested, swappable Python.

<!-- demo.gif goes here: 60 seconds of upload → ask → cited answer -->

## What it does

- **Ingest anything** — PDF, Word, PowerPoint, Excel, HTML, images (OCR) via
  [Docling](https://github.com/docling-project/docling), layout-aware (tables survive)
- **Structure-aware chunking** — chunks follow headings and pages, never split tables,
  and carry their provenance (file, page, heading)
- **State-of-the-art retrieval** — hybrid search (dense embeddings + BM25) fused with
  RRF in [Qdrant](https://qdrant.tech), then cross-encoder **reranking**
- **Grounded answers** — Claude answers *only* from retrieved passages and cites them
  inline; citations are parsed and returned as structured source references
- **Honest errors** — 503 when no model key is configured, 401 bad key, 429 rate
  limit, 502 upstream — never a mystery 500
- **Ops-ready** — API-key auth, health checks, Docker + nginx load balancing,
  Kubernetes manifests, CI

All retrieval runs on **free local models** (ONNX, CPU); only answer generation
calls the Claude API.

## Architecture

```
                    ┌────────────────── docker compose / k8s ───────────────┐
  Swagger /docs     │   ┌───────┐      ┌──────────┐      ┌─────────┐        │
  Dashboard /   ────┼──►│ nginx │─────►│ api  #1  │─────►│ Qdrant  │        │
  any client        │   │ LB+RL │  ╲   ├──────────┤      │ vectors │        │
                    │   └───────┘   ╲─►│ api  #2  │      └─────────┘        │
                    │                  └────┬─────┘                         │
                    └───────────────────────┼───────────────────────────────┘
                                            ▼ Anthropic API (answers only)
     rag_core:  Parser → Chunker → Embedder ─→ VectorStore
                Retriever (hybrid, RRF) → Reranker → LLM (cited answers)
```

The reusable logic lives in [`rag_core/`](rag_core/) — zero web code, every stage
behind a Protocol ([interfaces.py](rag_core/interfaces.py)), so any piece swaps
without touching the rest. [`app/`](app/) is a thin FastAPI shell.

## Eval results

Retrieval quality is **measured, not assumed** — 30 hand-labeled questions over the
sample corpus, run with `uv run python evals/run.py`:

| Configuration | recall@1 | recall@5 |
|---|---|---|
| dense only | 97% | 100% |
| hybrid (dense + BM25) | 93% | 100% |
| hybrid + cross-encoder rerank | **97%** | **100%** |

On this small, clean demo corpus the baseline is already strong and @5 saturates —
the honest reading, and why the harness matters: re-run it after any change to see
whether retrieval got better or worse. Details: [docs/concepts/11-evals.md](docs/concepts/11-evals.md).

## Quickstart

**Local dev** (Python 3.11 + [uv](https://docs.astral.sh/uv/)):

```
uv sync --dev
uv run uvicorn app.main:app --reload
```

- Dashboard: http://localhost:8000 (API key: `dev-key` until you change it)
- Swagger UI: http://localhost:8000/docs
- Set `ANTHROPIC_API_KEY` in `.env` to enable answers (retrieval works without it)

**The whole system with Docker** (nginx + 2 API replicas + Qdrant):

```
docker compose up --build --scale api=2
```

then watch the load balancing: `curl -i http://localhost:8080/health` twice —
the `X-Instance` header alternates between replicas.

**Kubernetes** (e.g. Docker Desktop's built-in cluster):

```
docker build -t rag-microservice:latest -f docker/Dockerfile .
kubectl apply -f k8s/
kubectl port-forward svc/rag-api 8080:8000
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload & index a document (multipart) |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{source}` | Remove a document and its chunks |
| `POST` | `/query` | Ask a question → answer + citations |
| `GET` | `/health` | Liveness (public) |

All data endpoints require the `X-API-Key` header — use the **Authorize** button
in Swagger. Configuration is pure environment variables ([.env.example](.env.example)).

## Tests & verification

```
uv run pytest          # 39 unit/integration tests, LLM mocked, in-memory Qdrant
uv run ruff check .    # lint
uv run python evals/run.py   # retrieval eval (local models, free)
```

CI runs lint + tests on every push.

## Concept notes

Every design decision is documented in [docs/concepts/](docs/concepts/) — RAG,
embeddings & hybrid search, reranking, grounding & citations, 12-factor config,
Docker & nginx, Kubernetes, evals. Written to be readable by non-specialists.

## Honest limitations

- The eval corpus is small (15 chunks); the pipeline's advantage over plain dense
  search shows on larger, noisier corpora.
- Embedded Qdrant mode is single-process (dev convenience); the compose/K8s setups
  use the real server.
- Answers stream as a single response (no token streaming yet) — see roadmap.

## Roadmap

- [ ] Token streaming for answers (SSE)
- [ ] BYOK mode: per-request `X-Anthropic-Key` header so visitors bring their own key
- [ ] Persistent model cache volume in K8s (PVC instead of emptyDir)
