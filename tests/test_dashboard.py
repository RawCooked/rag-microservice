"""Dashboard fragments: server-rendered HTML over the same pipeline."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from tests.fakes import FakePipeline

client = TestClient(app)
AUTH = {"X-API-Key": "dev-key"}


@pytest.fixture
def fake_pipeline():
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_dashboard_page_renders(fake_pipeline):
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Microservice" in response.text
    assert "htmx" in response.text


def test_documents_fragment_lists_sources(fake_pipeline):
    response = client.get("/ui/documents", headers=AUTH)
    assert response.status_code == 200
    assert "existing.pdf" in response.text


def test_query_fragment_renders_answer_with_citations(fake_pipeline):
    response = client.post(
        "/ui/query", data={"question": "What is the refund policy?"}, headers=AUTH
    )
    assert response.status_code == 200
    assert "30 days" in response.text
    assert "policy.pdf" in response.text


def test_query_fragment_rejects_empty_question(fake_pipeline):
    response = client.post("/ui/query", data={"question": " "}, headers=AUTH)
    assert response.status_code == 200
    assert "error" in response.text  # rendered error fragment
