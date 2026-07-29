# tests/test_api.py
import numpy as np
from fastapi.testclient import TestClient

import api.main as main_module


class _DummyModel:
    def predict_proba(self, X):
        return np.array([[0.9, 0.1]])


def _payload(amount: float = 100.0) -> dict:
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Amount"] = amount
    return payload


def test_health():
    client = TestClient(main_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metrics_returns_real_training_metrics():
    client = TestClient(main_module.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "pr_auc" in body
    assert "precision_fraud" in body


def test_predict_returns_probability(monkeypatch):
    monkeypatch.setattr(main_module, "MODEL", _DummyModel())
    client = TestClient(main_module.app)
    resp = client.post("/predict", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_probability"] == 0.1
    assert body["is_fraud"] is False


def test_predict_validates_negative_amount():
    client = TestClient(main_module.app)
    resp = client.post("/predict", json=_payload(amount=-5.0))
    assert resp.status_code == 422


def test_predict_rate_limit_blocks_excess_requests(monkeypatch):
    monkeypatch.setattr(main_module, "MODEL", _DummyModel())
    client = TestClient(main_module.app)
    statuses = [client.post("/predict", json=_payload()).status_code for _ in range(15)]
    assert 429 in statuses
