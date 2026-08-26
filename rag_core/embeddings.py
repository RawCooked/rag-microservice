"""Local embedding models via fastembed (ONNX — no GPU, no API key).

Two representations per text, one for each half of hybrid search:

- dense: a 384-dim semantic vector from a small open-source model.
  Similar meaning → nearby vectors, regardless of word overlap.
- sparse: BM25-style keyword weights. Exact-term matching — catches
  product codes, names, and acronyms that semantic vectors blur.

Both models download once (~100 MB total) and run on CPU. Swapping to
an embedding API later means writing one new class that satisfies the
`Embedder` protocol — nothing else changes.
"""

from __future__ import annotations

from rag_core.interfaces import SparseVector


class FastembedEmbedder:
    def __init__(
        self,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "Qdrant/bm25",
    ) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        self.dense_dim = len(next(iter(self._dense.embed(["probe"]))))

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._dense.embed(texts)]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in self._sparse.embed(texts)
        ]
