# api/main.py
"""Rate-limited FastAPI fraud-detection endpoint.

No external paid API dependency (the model is a local scikit-learn
artifact) and in-memory rate limiting (10 requests/minute/IP) so a public
deployment can't run up hosting costs under heavy or malicious traffic.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL = joblib.load(MODEL_DIR / "fraud_model.joblib")
METRICS = json.loads((MODEL_DIR / "metrics.json").read_text())
FEATURE_COLUMNS = METRICS["feature_columns"]

# A ~29-field JSON transaction is well under 1KB; 64KB is a generous ceiling
# that still rejects large bodies before they're fully parsed, protecting
# Render's free-tier 512MB RAM from a few concurrent oversized requests.
MAX_BODY_BYTES = 64 * 1024


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds MAX_BODY_BYTES
    with 413, before the body is read or the route handler (and rate
    limiter) runs."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request body exceeds {MAX_BODY_BYTES} byte limit."
                    },
                )
        return await call_next(request)


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Fraud Detection API",
    description="Real RandomForest classifier trained on the ULB Credit Card Fraud dataset.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(MaxBodySizeMiddleware)


def _sanitize_non_finite(value):
    """Recursively replace float inf/-inf/nan with their str() form.

    Rejecting inf/nan/-inf input (Transaction.model_config,
    allow_inf_nan=False) raises a RequestValidationError whose default
    FastAPI handler echoes the offending raw value back in the error
    body. Starlette's JSONResponse encodes with allow_nan=False (strict
    JSON), so without this sanitization step that echo itself blows up
    with an unhandled 500 instead of the intended clean 422.
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = jsonable_encoder(exc.errors())
    return JSONResponse(status_code=422, content={"detail": _sanitize_non_finite(errors)})


class Transaction(BaseModel):
    # extra="forbid" rejects unexpected/typo'd fields with 422 instead of
    # silently ignoring them. allow_inf_nan=False rejects Infinity/-Infinity
    # (and, in this pydantic version, NaN too) with a clean 422 instead of
    # letting them reach sklearn and cause a 500.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

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
    # Built as a named, single-row DataFrame (not a bare list-of-lists)
    # because the model was fit on a named pandas DataFrame: this avoids
    # sklearn's "X does not have valid feature names" warning and makes the
    # column order structurally tied to FEATURE_COLUMNS rather than
    # incidental.
    row = pd.DataFrame(
        [[getattr(transaction, col) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    proba = float(MODEL.predict_proba(row)[0][1])
    return PredictionResponse(fraud_probability=proba, is_fraud=proba >= 0.5)
