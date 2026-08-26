"""The pipeline: wires the pluggable stages together.

`RagPipeline` only knows the *protocols* from interfaces.py — it works
identically with the real Docling/fastembed/Qdrant stack or with fakes
in tests. `build_pipeline()` is the one place that chooses concrete
implementations (composition root), driven entirely by Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_core.config import Settings
from rag_core.interfaces import (
    Answer,
    AnswerGenerator,
    Chunker,
    Embedder,
    Parser,
    Reranker,
    ScoredChunk,
    VectorStore,
)


@dataclass
class IngestResult:
    source: str
    num_chunks: int
    num_pages: int | None


class RagPipeline:
    def __init__(
        self,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        answerer: AnswerGenerator | None = None,
        fetch_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.answerer = answerer
        self.fetch_k = fetch_k
        self.top_k = top_k

    def ingest(self, path: Path) -> IngestResult:
        doc = self.parser.parse(path)
        chunks = self.chunker.chunk(doc)
        texts = [chunk.text for chunk in chunks]
        dense = self.embedder.embed_dense(texts)
        sparse = self.embedder.embed_sparse(texts)
        # Remove any previous version of this document first, so
        # re-uploading a file replaces it instead of mixing versions.
        self.store.delete_source(doc.source)
        self.store.add(chunks, dense, sparse)
        return IngestResult(source=doc.source, num_chunks=len(chunks), num_pages=doc.num_pages)

    def list_sources(self) -> list[str]:
        return self.store.list_sources()

    def delete_source(self, source: str) -> None:
        self.store.delete_source(source)

    def retrieve(self, question: str) -> list[ScoredChunk]:
        """Two-stage retrieval: hybrid search (recall) then rerank (precision)."""
        query_dense = self.embedder.embed_dense([question])[0]
        query_sparse = self.embedder.embed_sparse([question])[0]
        candidates = self.store.hybrid_search(query_dense, query_sparse, self.fetch_k)
        if self.reranker is None:
            return candidates[: self.top_k]
        return self.reranker.rerank(question, candidates, self.top_k)

    def query(self, question: str) -> Answer:
        if self.answerer is None:
            raise RuntimeError("Pipeline was built without an answer generator")
        top = self.retrieve(question)
        return self.answerer.answer(question, [scored.chunk for scored in top])


def build_pipeline(settings: Settings) -> RagPipeline:
    """Composition root: assemble the production stack (heavy imports live here)."""
    from rag_core.chunking import MarkdownChunker
    from rag_core.embeddings import FastembedEmbedder
    from rag_core.llm import ClaudeAnswerGenerator
    from rag_core.parsing import DoclingParser
    from rag_core.retrieval import FastembedReranker
    from rag_core.store import QdrantStore

    embedder = FastembedEmbedder(
        dense_model=settings.dense_model, sparse_model=settings.sparse_model
    )
    return RagPipeline(
        parser=DoclingParser(),
        chunker=MarkdownChunker(),
        embedder=embedder,
        store=QdrantStore.from_settings(settings, dense_dim=embedder.dense_dim),
        reranker=FastembedReranker(model=settings.reranker_model),
        answerer=ClaudeAnswerGenerator(
            model=settings.anthropic_model, api_key=settings.anthropic_api_key
        ),
        fetch_k=settings.fetch_k,
        top_k=settings.top_k,
    )
