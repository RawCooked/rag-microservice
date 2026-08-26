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
from rag_core.interfaces import Chunker, Embedder, Parser, VectorStore


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
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

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


def build_pipeline(settings: Settings) -> RagPipeline:
    """Composition root: assemble the production stack (heavy imports live here)."""
    from rag_core.chunking import MarkdownChunker
    from rag_core.embeddings import FastembedEmbedder
    from rag_core.parsing import DoclingParser
    from rag_core.store import QdrantStore

    embedder = FastembedEmbedder(
        dense_model=settings.dense_model, sparse_model=settings.sparse_model
    )
    return RagPipeline(
        parser=DoclingParser(),
        chunker=MarkdownChunker(),
        embedder=embedder,
        store=QdrantStore.from_settings(settings, dense_dim=embedder.dense_dim),
    )
