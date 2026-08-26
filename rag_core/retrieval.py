"""Reranking: the second, precise stage of retrieval.

Hybrid search is fast because it compares pre-computed vectors — but
each chunk was embedded without knowing the question. A cross-encoder
reads the question and a candidate *together* and scores true
relevance. Too slow to run over the whole corpus; exactly right for
re-scoring the ~20 hybrid-search candidates down to the best ~5.

Runs locally via fastembed (ONNX, CPU, free) like the embedders.
"""

from __future__ import annotations

from rag_core.interfaces import ScoredChunk


class FastembedReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._encoder = TextCrossEncoder(model_name=model)

    def rerank(
        self, question: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = list(self._encoder.rerank(question, [c.chunk.text for c in candidates]))
        pairs = zip(candidates, scores, strict=True)
        ranked = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in ranked[:top_k]]
