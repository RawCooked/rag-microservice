"""Qdrant vector store with hybrid search.

Each chunk is stored with two named vectors — "dense" (semantic) and
"bm25" (sparse keyword weights). A hybrid query runs both searches and
merges them with Reciprocal Rank Fusion (RRF): each result is scored by
its *rank* in each list (1/rank, roughly), so the two scoring scales —
cosine similarity vs. BM25 — combine without any manual weight tuning.

Connection modes (chosen by config, not code — 12-factor):
- no URL  → embedded local mode, data in a folder. Dev and tests.
- URL set → real Qdrant server (docker-compose / Kubernetes).
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag_core.config import Settings
from rag_core.interfaces import Chunk, ScoredChunk, SparseVector

DENSE = "dense"
SPARSE = "bm25"


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dense_dim: int) -> None:
        self.client = client
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

    @classmethod
    def from_settings(cls, settings: Settings, dense_dim: int) -> QdrantStore:
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url)
        else:
            client = QdrantClient(path=settings.qdrant_path)
        return cls(client, settings.collection, dense_dim)

    def add(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> None:
        points = [
            models.PointStruct(
                id=chunk.id,
                vector={
                    DENSE: dense[i],
                    SPARSE: models.SparseVector(indices=sparse[i].indices, values=sparse[i].values),
                },
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "heading": chunk.heading,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: SparseVector,
        limit: int,
    ) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=query_dense, using=DENSE, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices, values=query_sparse.values
                    ),
                    using=SPARSE,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk=Chunk(
                    id=str(point.id),
                    text=point.payload["text"],
                    source=point.payload["source"],
                    page=point.payload.get("page"),
                    heading=point.payload.get("heading"),
                ),
                score=point.score,
            )
            for point in result.points
        ]

    def list_sources(self) -> list[str]:
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            sources.update(p.payload["source"] for p in points)
            if offset is None:
                return sorted(sources)

    def delete_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="source", match=models.MatchValue(value=source))
                    ]
                )
            ),
        )
