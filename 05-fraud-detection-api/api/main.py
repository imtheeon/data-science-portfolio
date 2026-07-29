# api/main.py
"""Rate-limited FastAPI fraud-detection endpoint.

No external paid API dependency (the model is a local scikit-learn
artifact) and in-memory rate limiting (10 requests/minute/IP) so a public
deployment can't run up hosting costs under heavy or malicious traffic.
"""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL = joblib.load(MODEL_DIR / "fraud_model.joblib")
METRICS = json.loads((MODEL_DIR / "metrics.json").read_text())
FEATURE_COLUMNS = METRICS["feature_columns"]

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Fraud Detection API",
    description="Real RandomForest classifier trained on the ULB Credit Card Fraud dataset.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return METRICS


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, transaction: Transaction):
    row = [[getattr(transaction, col) for col in FEATURE_COLUMNS]]
    proba = float(MODEL.predict_proba(row)[0][1])
    return PredictionResponse(fraud_probability=proba, is_fraud=proba >= 0.5)
