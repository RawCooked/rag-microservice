# Evals: measuring the AI, not just the code

Unit tests answer "does the code work?". They cannot answer "does the
*retrieval* actually find the right passage?" — that needs labeled
data and a metric. That's an **eval**.

## The dataset
30 hand-labeled questions over 3 sample documents
([questions.json](../../evals/questions.json)). Each label says which
document and which passage substring the retriever must surface. The
questions deliberately avoid copying the document's wording
("How long do I have to get my money back?" vs. "full refund within
30 days") so semantic search is actually exercised, and include
exact-token cases (error codes, part numbers) where BM25 shines.

## The metric
**recall@k** — how often the correct passage appears in the top k.
Measured for dense-only, hybrid, and hybrid+rerank, at k=1 and k=5,
so every pipeline stage has to justify itself against a baseline.

## Reading the numbers honestly
On this small demo corpus (15 chunks) the baseline is already strong
and recall@5 saturates — the interesting number is recall@1, and the
one miss is documented in the results. Hybrid + reranking earns its
keep on *larger, noisier* corpora; the eval harness is what lets you
verify that claim instead of asserting it. That framing — knowing what
your benchmark can and cannot show — is itself the skill.

## Why this matters
"How do you know your AI feature works, and how do you know when it
breaks?" is the interview question. This folder is the answer: change
the chunker, swap the embedder, re-run `uv run python evals/run.py`,
and the number tells you if you made things better or worse.
