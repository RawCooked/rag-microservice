import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from rag_core.llm import LLMNotConfiguredError
from tests.fakes import FakePipeline

client = TestClient(app)
AUTH = {"X-API-Key": "dev-key"}


@pytest.fixture
def fake_pipeline():
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_query_returns_answer_with_citations(fake_pipeline):
    response = client.post(
        "/query", json={"question": "What is the refund policy?"}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert "30 days" in body["answer"]
    assert body["model"] == "fake-model"
    assert body["citations"][0]["source"] == "policy.pdf"
    assert body["citations"][0]["marker"] == 1


def test_question_is_validated(fake_pipeline):
    assert client.post("/query", json={}, headers=AUTH).status_code == 422
    assert client.post("/query", json={"question": "ab"}, headers=AUTH).status_code == 422


def test_missing_anthropic_key_maps_to_503():
    class Unconfigured(FakePipeline):
        def query(self, question):
            raise LLMNotConfiguredError("No Anthropic API key configured.")

    app.dependency_overrides[get_pipeline] = lambda: Unconfigured()
    try:
        response = client.post(
            "/query", json={"question": "Anything at all?"}, headers=AUTH
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "API key" in response.json()["detail"]
