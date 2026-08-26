"""Retrieval eval: recall@k over a labeled question set.

Measures how often the correct passage is retrieved, comparing three
configurations so each pipeline stage has to justify itself:

- dense-only search
- hybrid search (dense + BM25, RRF fusion)
- hybrid + cross-encoder reranking  (the production path)

A question counts as a hit when a retrieved chunk comes from the
expected document AND contains the expected substring (whitespace-
normalized, case-insensitive).

Runs entirely on local models (free, no API key needed):

    uv run python evals/run.py

If ANTHROPIC_API_KEY is set, it additionally runs a small grounded-
answer check: every answer must cite at least one passage from the
expected document. Results are written to evals/results.md.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient

from rag_core.chunking import MarkdownChunker
from rag_core.config import get_settings
from rag_core.embeddings import FastembedEmbedder
from rag_core.interfaces import ScoredChunk, SparseVector
from rag_core.parsing import DoclingParser
from rag_core.pipeline import RagPipeline
from rag_core.retrieval import FastembedReranker
from rag_core.store import QdrantStore

EVALS_DIR = Path(__file__).parent
K = 5


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def is_hit(retrieved: list[ScoredChunk], source: str, must_contain: str) -> bool:
    needle = normalize(must_contain)
    return any(
        scored.chunk.source == source and needle in normalize(scored.chunk.text)
        for scored in retrieved
    )


def dense_only_search(store: QdrantStore, dense: list[float], limit: int) -> list[ScoredChunk]:
    # Same store, dense vector only — the baseline hybrid search must beat.
    empty_sparse = SparseVector(indices=[], values=[])
    return store.hybrid_search(dense, empty_sparse, limit)


def main() -> None:
    questions = json.loads((EVALS_DIR / "questions.json").read_text(encoding="utf-8"))

    print("Loading local models (first run downloads them)...")
    embedder = FastembedEmbedder()
    reranker = FastembedReranker()
    store = QdrantStore(QdrantClient(":memory:"), collection="eval", dense_dim=embedder.dense_dim)
    pipeline = RagPipeline(
        parser=DoclingParser(),
        chunker=MarkdownChunker(),
        embedder=embedder,
        store=store,
        reranker=reranker,
        fetch_k=20,
        top_k=K,
    )

    for path in sorted((EVALS_DIR / "documents").glob("*.md")):
        result = pipeline.ingest(path)
        print(f"  indexed {result.source}: {result.num_chunks} chunks")

    configs = ["dense@1", "hybrid@1", "rerank@1", "dense@5", "hybrid@5", "rerank@5"]
    hits = dict.fromkeys(configs, 0)
    misses: list[tuple[str, str]] = []

    for item in questions:
        question, source, needle = item["question"], item["source"], item["must_contain"]
        dense = embedder.embed_dense([question])[0]
        sparse = embedder.embed_sparse([question])[0]

        dense_top = dense_only_search(store, dense, K)
        hybrid_top20 = store.hybrid_search(dense, sparse, 20)
        reranked = reranker.rerank(question, hybrid_top20, K)

        hits["dense@1"] += is_hit(dense_top[:1], source, needle)
        hits["hybrid@1"] += is_hit(hybrid_top20[:1], source, needle)
        hits["rerank@1"] += is_hit(reranked[:1], source, needle)
        hits["dense@5"] += is_hit(dense_top, source, needle)
        hits["hybrid@5"] += is_hit(hybrid_top20[:K], source, needle)
        hits["rerank@5"] += is_hit(reranked, source, needle)
        if not is_hit(reranked[:1], source, needle):
            misses.append((question, "rerank@1"))

    n = len(questions)
    print(f"\nRetrieval recall over {n} labeled questions:")
    for config in configs:
        print(f"  {config:<10} {hits[config]}/{n}  ({hits[config] / n:.0%})")
    for question, config in misses:
        print(f"  MISS [{config}]: {question}")

    # Optional grounded-answer check (needs an Anthropic API key).
    answer_line = "not run (no ANTHROPIC_API_KEY set)"
    if os.environ.get("ANTHROPIC_API_KEY"):
        from rag_core.llm import ClaudeAnswerGenerator

        model = get_settings().anthropic_model
        print(f"\nGrounded-answer check with {model} (first 10 questions)...")
        pipeline.answerer = ClaudeAnswerGenerator(model=model)
        cited_ok = 0
        sample = questions[:10]
        for item in sample:
            answer = pipeline.query(item["question"])
            ok = any(c.chunk.source == item["source"] for c in answer.citations)
            cited_ok += ok
            print(f"  {'ok ' if ok else 'BAD'} {item['question']}")
        answer_line = f"{cited_ok}/{len(sample)} answers cite the correct document ({model})"
        print(f"  -> {answer_line}")

    results_md = "\n".join(
        [
            "# Eval results",
            "",
            f"_Last run: {date.today().isoformat()} — {n} labeled questions, "
            "local models (bge-small + BM25 + ms-marco cross-encoder)._",
            "",
            "| Configuration | Recall | Score |",
            "|---|---|---|",
            *(
                f"| {config} | {hits[config] / n:.0%} | {hits[config]}/{n} |"
                for config in configs
            ),
            "",
            f"Grounded answers: {answer_line}",
            "",
        ]
    )
    (EVALS_DIR / "results.md").write_text(results_md, encoding="utf-8")
    print(f"\nWrote {EVALS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
