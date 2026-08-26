"""QdrantStore tests against Qdrant's in-memory mode — real hybrid
search logic (dense + sparse + RRF fusion), zero servers."""

import pytest
from qdrant_client import QdrantClient

from rag_core.interfaces import SparseVector
from rag_core.store import QdrantStore
from tests.fakes import make_chunk


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), collection="test", dense_dim=4)


def populated(store: QdrantStore) -> QdrantStore:
    chunks = [
        make_chunk(1, "refund policy thirty days", source="policy.pdf"),
        make_chunk(2, "shipping takes one week", source="policy.pdf"),
        make_chunk(3, "the API rate limit is 100", source="api.md"),
    ]
    dense = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    sparse = [
        SparseVector(indices=[10], values=[1.0]),
        SparseVector(indices=[20], values=[1.0]),
        SparseVector(indices=[30], values=[1.0]),
    ]
    store.add(chunks, dense, sparse)
    return store


def test_hybrid_search_ranks_matching_chunk_first(store):
    populated(store)
    # query vector aligned with chunk 1 (dense) AND its sparse keyword
    results = store.hybrid_search(
        query_dense=[1.0, 0.0, 0.0, 0.0],
        query_sparse=SparseVector(indices=[10], values=[1.0]),
        limit=3,
    )
    assert results
    assert results[0].chunk.text == "refund policy thirty days"
    assert results[0].chunk.source == "policy.pdf"


def test_sparse_signal_alone_can_win():
    # dense query matches nothing; sparse keyword points at chunk 3
    fresh = QdrantStore(QdrantClient(":memory:"), collection="test", dense_dim=4)
    populated(fresh)
    results = fresh.hybrid_search(
        query_dense=[0.0, 0.0, 0.0, 1.0],
        query_sparse=SparseVector(indices=[30], values=[1.0]),
        limit=1,
    )
    assert results[0].chunk.source == "api.md"


def test_list_sources_and_delete_source(store):
    populated(store)
    assert store.list_sources() == ["api.md", "policy.pdf"]
    store.delete_source("policy.pdf")
    assert store.list_sources() == ["api.md"]
