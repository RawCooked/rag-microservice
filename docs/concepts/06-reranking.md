# Reranking: retrieval's second stage

Hybrid search is **fast but approximate**: every chunk was embedded
*before* any question existed, so its vector is a general-purpose
summary. Cost: one vector comparison per chunk — cheap enough to scan
everything.

A **cross-encoder** is the opposite trade-off: it reads the question
and one candidate *together* through a full transformer and outputs a
relevance score for exactly that pair. Far more accurate, far too slow
to run on the whole corpus.

So retrieval becomes a funnel:

```
whole corpus ──hybrid search──► ~20 candidates ──cross-encoder──► top 5
   (milliseconds, approximate)          (a second, precise)
```

The eval in phase 7 measures recall with and without the reranker —
the published numbers show it earning its latency.

Both stages run locally (fastembed, ONNX, CPU): retrieval happens on
every query, so putting it on a paid API would make every visitor cost
money. Only the final answer generation uses the Claude API.
