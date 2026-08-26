from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "instance" in body


def test_every_response_carries_instance_header():
    response = client.get("/health")
    assert "X-Instance" in response.headers


def test_swagger_ui_is_served():
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
