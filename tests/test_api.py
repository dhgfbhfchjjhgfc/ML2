import pytest
from starlette.testclient import TestClient
from mlproject.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_custom_predict_invalid_amount_returns_422(client):
    """Свой тест 1: отрицательная сумма возвращает 422 Unprocessable Entity."""
    payload = {"Amount": -100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

def test_custom_predict_forbidden_extra_field_returns_422(client):
    """Свой тест 2: передача незарегистрированного поля возвращает 422."""
    payload = {"Amount": 50.0, "unknown_field": "test"}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
