"""ClaudeAnswerGenerator tests with a stubbed Anthropic client —
citation parsing, refusal handling, and model routing, no network."""

from types import SimpleNamespace

from rag_core.llm import ClaudeAnswerGenerator
from tests.fakes import make_chunk


class StubClient:
    """Mimics the two SDK surfaces the generator may call."""

    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []
        self.messages = SimpleNamespace(create=self._plain_create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta_create))

    def _plain_create(self, **kwargs):
        self.calls.append(("plain", kwargs))
        return self.response

    def _beta_create(self, **kwargs):
        self.calls.append(("beta", kwargs))
        return self.response


def text_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="stub-model",
    )


CHUNKS = [make_chunk(i, f"passage number {i}") for i in range(1, 4)]


def test_citations_are_parsed_and_out_of_range_markers_dropped():
    client = StubClient(text_response("Thirty days [1][3]. Unrelated claim [9]."))
    generator = ClaudeAnswerGenerator(model="claude-haiku-4-5", client=client)
    answer = generator.answer("How long?", CHUNKS)
    assert [c.marker for c in answer.citations] == [1, 3]
    assert answer.citations[0].chunk.text == "passage number 1"
    assert answer.model == "stub-model"


def test_prompt_contains_numbered_chunks_and_question():
    client = StubClient(text_response("Answer [1]."))
    ClaudeAnswerGenerator(model="claude-haiku-4-5", client=client).answer("Why?", CHUNKS)
    kind, kwargs = client.calls[0]
    assert kind == "plain"
    prompt = kwargs["messages"][0]["content"]
    assert "[1] doc.pdf" in prompt
    assert "passage number 3" in prompt
    assert "Question: Why?" in prompt


def test_opus5_routes_through_beta_with_fallbacks():
    client = StubClient(text_response("Answer [2]."))
    ClaudeAnswerGenerator(model="claude-opus-5", client=client).answer("Q?", CHUNKS)
    kind, kwargs = client.calls[0]
    assert kind == "beta"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_refusal_returns_graceful_answer():
    client = StubClient(text_response("", stop_reason="refusal"))
    answer = ClaudeAnswerGenerator(model="claude-haiku-4-5", client=client).answer("Q?", CHUNKS)
    assert answer.citations == []
    assert "declined" in answer.text


def test_missing_credentials_raise_llm_not_configured():
    import pytest

    from rag_core.llm import LLMNotConfiguredError

    class NoAuthClient(StubClient):
        def _plain_create(self, **kwargs):
            raise TypeError("Could not resolve authentication method.")

    generator = ClaudeAnswerGenerator(model="claude-haiku-4-5", client=NoAuthClient(None))
    with pytest.raises(LLMNotConfiguredError):
        generator.answer("Q?", CHUNKS)
