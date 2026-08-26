"""Answer generation with Claude, grounded in retrieved chunks.

The model receives the question plus the top retrieved passages,
numbered [1]..[n], and is instructed to answer ONLY from them and to
cite the passage numbers inline. We then parse those markers back into
structured citations pointing at source file + page — the verifiable
part of every answer.
"""

from __future__ import annotations

import re

from rag_core.interfaces import Answer, Chunk, Citation

GROUNDING_SYSTEM = """You answer questions using ONLY the numbered context passages provided.

Rules:
- Base every claim on the passages. If they do not contain the answer, say so plainly — never guess.
- Cite the supporting passage numbers inline in square brackets, e.g. [1] or [2][3],
  right after each claim.
- Be concise and direct. Answer in the language of the question."""

# Server-side refusal fallbacks exist only on these models; on a safety
# decline the API transparently retries on a fallback model.
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5"}

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


class LLMNotConfiguredError(Exception):
    """Raised when no Anthropic credentials are available."""


class ClaudeAnswerGenerator:
    def __init__(self, model: str, api_key: str = "", client=None) -> None:
        self.model = model
        self._api_key = api_key
        self._client = client  # injectable for tests

    def _get_client(self):
        if self._client is None:
            import anthropic

            try:
                self._client = anthropic.Anthropic(api_key=self._api_key or None)
            except anthropic.AnthropicError as exc:
                raise LLMNotConfiguredError(
                    "No Anthropic API key configured — set ANTHROPIC_API_KEY."
                ) from exc
        return self._client

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        client = self._get_client()
        context = "\n\n".join(
            f"[{i}] {chunk.source}"
            + (f", page {chunk.page}" if chunk.page else "")
            + f"\n{chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        )
        request = dict(
            model=self.model,
            max_tokens=4096,
            system=GROUNDING_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        try:
            if self.model in _FALLBACK_MODELS:
                response = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = client.messages.create(**request)
        except TypeError as exc:
            # The SDK resolves credentials (env vars, login profiles) at
            # request time; with none available it raises TypeError here.
            if "authentication" in str(exc).lower():
                raise LLMNotConfiguredError(
                    "No Anthropic API key configured — set ANTHROPIC_API_KEY."
                ) from exc
            raise

        if response.stop_reason == "refusal":
            return Answer(
                text="The model declined to answer this question.",
                citations=[],
                model=response.model,
            )

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        markers = sorted({int(m) for m in _MARKER_RE.findall(text)})
        citations = [
            Citation(marker=m, chunk=chunks[m - 1]) for m in markers if 1 <= m <= len(chunks)
        ]
        return Answer(text=text, citations=citations, model=response.model)
