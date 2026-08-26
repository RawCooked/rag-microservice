import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from tests.fakes import FakePipeline


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_UPLOADS_DIR", str(tmp_path / "uploads"))
    # settings are cached — reset so the env var above is picked up
    from rag_core.config import get_settings

    get_settings.cache_clear()
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    yield fake
    app.dependency_overrides.clear()
    get_settings.cache_clear()


client = TestClient(app)
AUTH = {"X-API-Key": "dev-key"}


def test_upload_indexes_document(fake_pipeline):
    response = client.post(
        "/documents",
        files={"file": ("report.md", b"# Title\n\nSome content", "text/markdown")},
        headers=AUTH,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "report.md"
    assert body["chunks_indexed"] == 3
    assert len(fake_pipeline.ingested) == 1


def test_unsupported_extension_is_rejected(fake_pipeline):
    response = client.post(
        "/documents",
        files={"file": ("virus.exe", b"nope", "application/octet-stream")},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert fake_pipeline.ingested == []


def test_list_documents(fake_pipeline):
    response = client.get("/documents", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"documents": ["existing.pdf"]}


def test_delete_document(fake_pipeline):
    response = client.delete("/documents/existing.pdf", headers=AUTH)
    assert response.status_code == 204
    assert fake_pipeline.deleted == ["existing.pdf"]
