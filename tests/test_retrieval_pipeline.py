"""End-to-end pipeline test with fakes + real in-memory Qdrant:
ingest → retrieve → answer, no models, no network."""

from pathlib import Path

from qdrant_client import QdrantClient

from rag_core.chunking import MarkdownChunker
from rag_core.pipeline import RagPipeline
from rag_core.store import QdrantStore
from tests.fakes import FakeAnswerer, FakeEmbedder, FakeParser, FakeReranker


def build_test_pipeline() -> tuple[RagPipeline, FakeReranker, FakeAnswerer]:
    reranker, answerer = FakeReranker(), FakeAnswerer()
    pipeline = RagPipeline(
        parser=FakeParser(),
        chunker=MarkdownChunker(),
        embedder=FakeEmbedder(),
        store=QdrantStore(QdrantClient(":memory:"), collection="test", dense_dim=4),
        reranker=reranker,
        answerer=answerer,
        fetch_k=10,
        top_k=2,
    )
    return pipeline, reranker, answerer


def test_ingest_then_query_flows_through_all_stages():
    pipeline, reranker, answerer = build_test_pipeline()

    result = pipeline.ingest(Path("notes.md"))
    assert result.num_chunks >= 1
    assert pipeline.list_sources() == ["notes.md"]

    answer = pipeline.query("What does the doc say?")
    assert reranker.calls == ["What does the doc say?"]  # reranker was applied
    assert len(answerer.received) == 1
    assert len(answerer.received[0]) >= 1  # answerer got retrieved chunks
    assert answer.citations
    assert answer.citations[0].chunk.source == "notes.md"


def test_reingest_replaces_instead_of_duplicating():
    pipeline, _, _ = build_test_pipeline()
    first = pipeline.ingest(Path("notes.md"))
    second = pipeline.ingest(Path("notes.md"))
    assert first.num_chunks == second.num_chunks
    assert pipeline.list_sources() == ["notes.md"]
