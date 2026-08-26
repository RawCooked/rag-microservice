"""Every data endpoint requires X-API-Key; ops endpoints stay open."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from tests.fakes import FakePipeline

client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_pipeline():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/documents"),
        ("delete", "/documents/x.pdf"),
        ("get", "/ui/documents"),
    ],
)
def test_missing_key_is_rejected(method, path):
    assert getattr(client, method)(path).status_code == 401


def test_wrong_key_is_rejected():
    response = client.get("/documents", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_query_requires_key():
    response = client.post("/query", json={"question": "Is this open?"})
    assert response.status_code == 401


def test_health_and_docs_stay_public():
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/").status_code == 200
