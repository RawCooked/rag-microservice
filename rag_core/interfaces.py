"""Contracts between pipeline stages.

Every stage of the RAG pipeline is defined here as a Protocol
(structural typing: any class with matching methods satisfies it —
no inheritance required). The rest of the codebase depends only on
these contracts, which is what makes each part swappable: a local
embedding model and a cloud embedding API are interchangeable as
long as both look like `Embedder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Marker the parser inserts between pages of the source document, so the
# chunker can attribute each chunk to a page (needed for citations).
PAGE_BREAK = "<!-- page-break -->"


# --------------------------------------------------------------------------
# Data types passed between stages
# --------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """A document after parsing, before chunking."""

    source: str  # original filename
    markdown: str  # full text as markdown (tables preserved)
    num_pages: int | None = None


@dataclass
class Chunk:
    """One retrievable piece of a document, with provenance metadata."""

    id: str
    text: str
    source: str  # filename it came from
    page: int | None = None
    heading: str | None = None


@dataclass
class SparseVector:
    """Sparse (keyword/BM25-style) representation: only non-zero dims."""

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    """Links one answer reference [n] back to its source chunk."""

    marker: int  # the [n] used in the answer text
    chunk: Chunk


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model: str


# --------------------------------------------------------------------------
# Pipeline stage contracts
# --------------------------------------------------------------------------


class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


class Chunker(Protocol):
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


class Embedder(Protocol):
    """Produces dense (semantic) and sparse (keyword) vectors."""

    def embed_dense(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...


class VectorStore(Protocol):
    def add(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> None: ...

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: SparseVector,
        limit: int,
    ) -> list[ScoredChunk]: ...

    def list_sources(self) -> list[str]: ...

    def delete_source(self, source: str) -> None: ...


class Reranker(Protocol):
    def rerank(
        self, question: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class AnswerGenerator(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...
