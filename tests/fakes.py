"""Fake implementations of the rag_core protocols for fast, free tests.

Because every pipeline stage is a Protocol, these tiny classes are
drop-in replacements — no network, no models, fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

from rag_core.interfaces import Answer, Chunk, Citation, ParsedDocument, ScoredChunk, SparseVector
from rag_core.pipeline import IngestResult


class FakeParser:
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(source=path.name, markdown="# Doc\n\nHello world", num_pages=1)


class FakeEmbedder:
    """Maps each text to a trivial 4-dim vector. Deterministic."""

    dense_dim = 4

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        # One "keyword" per text: its length, weighted 1.0
        return [SparseVector(indices=[len(t) % 50], values=[1.0]) for t in texts]


class FakeReranker:
    """Reverses the candidate order — proves in tests that it was applied."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rerank(
        self, question: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        self.calls.append(question)
        return list(reversed(candidates))[:top_k]


class FakeAnswerer:
    """Echoes the chunks it received — no API, fully deterministic."""

    def __init__(self) -> None:
        self.received: list[list[Chunk]] = []

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        self.received.append(chunks)
        citations = [Citation(marker=i + 1, chunk=c) for i, c in enumerate(chunks)]
        return Answer(text=f"Answer to: {question} [1]", citations=citations, model="fake-model")


class FakePipeline:
    """Records calls; returns canned results. Used to test the API layer."""

    def __init__(self) -> None:
        self.ingested: list[Path] = []
        self.deleted: list[str] = []
        self.sources: list[str] = ["existing.pdf"]

    def ingest(self, path: Path) -> IngestResult:
        self.ingested.append(path)
        return IngestResult(source=path.name, num_chunks=3, num_pages=2)

    def list_sources(self) -> list[str]:
        return self.sources

    def delete_source(self, source: str) -> None:
        self.deleted.append(source)

    def query(self, question: str) -> Answer:
        chunk = make_chunk(1, "refunds within 30 days", source="policy.pdf")
        return Answer(
            text="Refunds are accepted within 30 days [1].",
            citations=[Citation(marker=1, chunk=chunk)],
            model="fake-model",
        )


def make_chunk(i: int, text: str, source: str = "doc.pdf") -> Chunk:
    return Chunk(id=f"00000000-0000-0000-0000-{i:012d}", text=text, source=source, page=1)
