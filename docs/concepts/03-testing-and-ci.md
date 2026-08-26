# Testing AI systems + CI

## The problem
LLM calls are slow, cost money, and give different output each run —
useless properties for automated tests.

## The two-layer answer

1. **Unit tests with mocks** (this repo's `tests/`): replace the LLM and
   embedder with fakes that return fixed values, then test *our* logic —
   parsing, chunking, retrieval math, API validation, auth — fast,
   free, and deterministic. These run on every push.

2. **Evals** (`evals/`, added in phase 7): a labeled dataset of
   question → correct-chunk pairs, scored automatically (recall@k).
   This measures the *quality* of the AI pipeline, separate from the
   *correctness* of the code. Run occasionally, results published in
   the README.

Code correctness and AI quality are different questions; strong AI
engineering measures both.

## CI (GitHub Actions)
On every push, GitHub boots a clean Linux machine, installs the project
with `uv sync`, and runs `ruff` (lint) + `pytest`. If anything fails the
commit is marked red. The green badge on the README is the public proof
that the project holds itself to a standard.
