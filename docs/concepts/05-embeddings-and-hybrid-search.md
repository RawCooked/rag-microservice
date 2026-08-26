# Embeddings, vector search, and why "hybrid"

## Dense vectors (semantic)
An embedding model maps text to ~384 numbers such that similar meaning
lands nearby: "reset my password" ≈ "credential recovery". Searching =
"find stored vectors nearest to the query vector" (cosine similarity).
That's what a **vector database** (Qdrant) is optimized for. We run a
small open-source model locally via fastembed (ONNX, CPU, no API key).

## Sparse vectors (keywords / BM25)
Semantic vectors blur exact tokens: product codes, acronyms, error
strings ("ERR_QUIC_PROTOCOL"), names. **BM25** — the classic search
algorithm — scores exact term matches, weighted by how rare each term
is (IDF). We store it as a *sparse* vector: only the terms that occur,
with weights.

## Hybrid search + RRF
Every chunk is stored with both vectors. A query runs **both searches**
and merges the two ranked lists with **Reciprocal Rank Fusion**: each
chunk is scored by its *rank* in each list (roughly 1/rank, summed).
Because ranks are scale-free, we don't have to tune "how much is 0.83
cosine worth vs. 12.4 BM25" — the standard, robust default.

## Why local models here
Retrieval runs on every query and every ingested chunk — putting it on
a paid API would make each visitor cost money. Locally it's free, fast,
and works offline; the `Embedder` protocol means a cloud embedding API
is a drop-in swap if quality ever demands it.
