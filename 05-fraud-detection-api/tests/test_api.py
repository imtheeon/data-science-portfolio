# tests/test_api.py
import json

import numpy as np
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import MAX_BODY_BYTES


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


def test_predict_with_real_model_produces_valid_probability():
    """No monkeypatching MODEL here — this exercises the actual trained
    fraud_model.joblib artifact end-to-end, since 'real model' is the
    project's central claim and no other test verifies it produces sane
    output."""
    client = TestClient(main_module.app)
    resp = client.post("/predict", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_rejects_infinity():
    client = TestClient(main_module.app)
    payload = _payload()
    payload["V1"] = float("inf")
    # httpx's `json=` kwarg refuses to serialize non-finite floats client
    # side (strict JSON), so send raw bytes via json.dumps's permissive
    # default (allow_nan=True) instead, to exercise server-side rejection.
    body = json.dumps(payload)
    resp = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_predict_rejects_nan():
    client = TestClient(main_module.app)
    payload = _payload()
    payload["V1"] = float("nan")
    body = json.dumps(payload)
    resp = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_predict_rejects_unexpected_fields():
    client = TestClient(main_module.app)
    payload = _payload()
    payload["not_a_real_field"] = 1.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_oversized_body():
    client = TestClient(main_module.app)
    payload = _payload()
    # Pad well past MAX_BODY_BYTES with a field that gets rejected by
    # extra="forbid" anyway — the point is the 413 must fire before that
    # validation ever runs, based on Content-Length alone.
    payload["padding"] = "x" * (MAX_BODY_BYTES + 1024)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 413


def test_predict_rate_limit_blocks_excess_requests(monkeypatch):
    monkeypatch.setattr(main_module, "MODEL", _DummyModel())
    client = TestClient(main_module.app)
    statuses = [client.post("/predict", json=_payload()).status_code for _ in range(15)]
    assert 429 in statuses
