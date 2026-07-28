# Project 5: Fraud Detection Live API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `05-fraud-detection-api/` in the `data-science-portfolio` repo: a real fraud classifier trained on the real ULB Credit Card Fraud dataset, wrapped in a rate-limited FastAPI endpoint, ready to deploy to Render — but not actually deployed until Leanthel approves.

**Architecture:** A training script that downloads the real, severely-imbalanced fraud dataset and trains a class-weighted RandomForest, saving the model and its real evaluation metrics to disk. A separate, minimal-dependency FastAPI app loads that saved model and exposes a rate-limited `/predict` endpoint plus `/health` and `/metrics`. A Dockerfile and `render.yaml` make the API deployable, with `autoDeploy: false` so nothing goes live without an explicit action.

**Tech Stack:** Python 3.12, pandas, scikit-learn, FastAPI, uvicorn, slowapi (in-memory rate limiting), pydantic, joblib, pytest, httpx (FastAPI TestClient), Docker, Kaggle API.

## Global Constraints

- No fabricated data or metrics: every number in the README comes from code in this repo actually being run. (Spec: "Hard requirement")
- Retrains a real model — does not reuse `data-analytics-portfolio`'s fraud dashboard, which has no real model behind it. (Spec: Project 5 decision)
- Report precision/recall/PR-AUC honestly; plain accuracy is not the headline metric on this severely imbalanced dataset (~0.17% fraud rate). (Spec: Project 5)
- No paid API dependency — the model is a local scikit-learn artifact, nothing to leak, no per-call cost. (Spec: "Guardrails")
- Rate limiting required on the public endpoint (e.g. 10 requests/minute/IP) so it can't run up hosting/API costs under traffic. (Spec: "Guardrails")
- **Nothing is actually deployed to Render, and nothing is pushed to GitHub, until Leanthel reviews and explicitly approves.** This plan builds and locally verifies everything up to (but not including) the live deploy action. (Spec: "Publishing gate")
- Working directory for all tasks: `~/data-science-portfolio/05-fraud-detection-api/`.

---

### Task 1: Real dataset acquisition and model training

**Files:**
- Create: `05-fraud-detection-api/requirements-train.txt`
- Create: `05-fraud-detection-api/train_model.py`

**Interfaces:**
- Consumes: Kaggle API credentials at `~/.kaggle/kaggle.json`.
- Produces: `model/fraud_model.joblib` (a fitted `sklearn.ensemble.RandomForestClassifier`) and `model/metrics.json` (keys: `pr_auc`, `precision_fraud`, `recall_fraud`, `f1_fraud`, `n_test`, `n_test_fraud`, `feature_columns`) — consumed by Task 2's API and Task 4's README.

- [ ] **Step 1: Create the project structure and venv**

```bash
cd ~/data-science-portfolio/05-fraud-detection-api
mkdir -p data model api tests
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Write `requirements-train.txt` and install**

```
pandas==2.2.2
scikit-learn==1.5.1
joblib==1.4.2
kaggle==1.6.17
```

```bash
pip install -r requirements-train.txt
```

- [ ] **Step 3: Confirm the exact Kaggle dataset ref**

Run: `kaggle datasets list -s "credit card fraud" --csv | head -10`
Confirm `mlg-ulb/creditcardfraud` (the well-known real ULB dataset: 284,807 real anonymized European credit card transactions, 492 real frauds, ~0.172% fraud rate, features `V1`-`V28` (PCA-anonymized) + `Amount` + `Class`) appears. Use the actual ref found if it differs.

- [ ] **Step 4: Write `train_model.py`**

```python
# train_model.py
"""Trains a fraud classifier on the real ULB Credit Card Fraud dataset.

Real dataset (Kaggle: mlg-ulb/creditcardfraud) — anonymized real European
credit card transactions from September 2013, ~0.172% fraud rate. Not
simulated.

Handles the severe class imbalance via class_weight="balanced" rather than
oversampling. Reports precision/recall/PR-AUC — plain accuracy is
misleading when ~99.8% of transactions are the majority (legitimate) class.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import kaggle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"  # confirmed in Task 1 Step 3

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def download() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    existing = list(DATA_DIR.glob("*.csv"))
    if existing:
        return existing[0]
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    downloaded = list(DATA_DIR.glob("*.csv"))
    if not downloaded:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR} after download.")
    return downloaded[0]


def main() -> None:
    df = pd.read_csv(download())
    n_fraud = int(df["Class"].sum())
    print(
        f"Loaded {len(df):,} real transactions, {n_fraud} real frauds "
        f"({df['Class'].mean() * 100:.4f}% fraud rate)."
    )

    X = df[FEATURE_COLUMNS]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_pred_proba)
    report = classification_report(
        y_test, y_pred, target_names=["legitimate", "fraud"], output_dict=True
    )
    print(f"PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "fraud"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "pr_auc": pr_auc,
        "precision_fraud": report["fraud"]["precision"],
        "recall_fraud": report["fraud"]["recall"],
        "f1_fraud": report["fraud"]["f1-score"],
        "n_test": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
        "feature_columns": FEATURE_COLUMNS,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}, metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it against the real dataset**

Run: `cd ~/data-science-portfolio/05-fraud-detection-api && source venv/bin/activate && python train_model.py`
Expected: real printed transaction/fraud counts, PR-AUC, and a full classification report. `model/fraud_model.joblib` and `model/metrics.json` both exist afterward. Save this console output — it is the source of Task 4's README numbers.

- [ ] **Step 6: Commit**

```bash
cd ~/data-science-portfolio
git add 05-fraud-detection-api/requirements-train.txt 05-fraud-detection-api/train_model.py 05-fraud-detection-api/model/metrics.json
git commit -m "Project 5: real fraud model trained on ULB dataset"
```

Note: `model/fraud_model.joblib` is a binary artifact — commit it too (`git add -f` is not needed, it isn't gitignored) since the deployed API loads it directly rather than retraining on every deploy:

```bash
git add 05-fraud-detection-api/model/fraud_model.joblib
git commit -m "Project 5: commit trained model artifact"
```

---

### Task 2: Rate-limited FastAPI app

**Files:**
- Create: `05-fraud-detection-api/requirements.txt`
- Create: `05-fraud-detection-api/api/__init__.py`
- Create: `05-fraud-detection-api/api/main.py`
- Create: `05-fraud-detection-api/tests/__init__.py`
- Create: `05-fraud-detection-api/tests/test_api.py`

**Interfaces:**
- Consumes: `model/fraud_model.joblib`, `model/metrics.json` (Task 1 — the API module loads these at import time, so Task 1 must run before this task's tests can pass).
- Produces: `api.main.app` (a `fastapi.FastAPI` instance with `GET /health`, `GET /metrics`, `POST /predict`), used by Task 3 (Dockerfile) and deployment.

- [ ] **Step 1: Write `requirements.txt`** (runtime-only, kept minimal since this is what ships in the deployed container — no `pandas`, no `kaggle`)

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
scikit-learn==1.5.1
joblib==1.4.2
slowapi==0.1.9
pydantic==2.9.2
```

```bash
pip install -r requirements.txt
pip install pytest==8.3.2 httpx==0.27.2
```

- [ ] **Step 2: Write the failing tests**

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd ~/data-science-portfolio/05-fraud-detection-api && source venv/bin/activate && python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'` (or import error if `model/fraud_model.joblib` from Task 1 is missing — if so, run Task 1 first).

- [ ] **Step 4: Implement `api/main.py`**

```python
# api/main.py
"""Rate-limited FastAPI fraud-detection endpoint.

No external paid API dependency (the model is a local scikit-learn
artifact) and in-memory rate limiting (10 requests/minute/IP) so a public
deployment can't run up hosting costs under heavy or malicious traffic.
"""

from __future__ import annotations

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
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6: Run the API locally and hit it for real with `curl`**

Run: `uvicorn api.main:app --port 8000 &` then:
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics
```
kill the background process afterward (`kill %1`).
Expected: real JSON responses (the `/metrics` output should match the real numbers from Task 1's `model/metrics.json`).

- [ ] **Step 7: Commit**

```bash
cd ~/data-science-portfolio
git add 05-fraud-detection-api/requirements.txt 05-fraud-detection-api/api 05-fraud-detection-api/tests
git commit -m "Project 5: rate-limited FastAPI fraud endpoint"
```

---

### Task 3: Dockerfile and Render deployment config (build, do not deploy)

**Files:**
- Create: `05-fraud-detection-api/Dockerfile`
- Create: `05-fraud-detection-api/.dockerignore`
- Create: `05-fraud-detection-api/render.yaml`

**Interfaces:**
- Consumes: `api/main.py`, `model/fraud_model.joblib`, `model/metrics.json`, `requirements.txt` (Task 1 & 2).
- Produces: a buildable Docker image; consumed only by the (later, gated) manual deploy step.

- [ ] **Step 1: Write `.dockerignore`**

```
venv/
data/
__pycache__/
*.pyc
.pytest_cache/
tests/
requirements-train.txt
train_model.py
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `render.yaml`**

```yaml
services:
  - type: web
    name: fraud-detection-api
    env: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
    plan: free
    healthCheckPath: /health
    autoDeploy: false
```

`autoDeploy: false` is deliberate — connecting this repo to Render does not go live on its own; an explicit deploy action is still required. This is the mechanism enforcing the "nothing deployed without approval" gate at the infra level, not just a process reminder.

- [ ] **Step 4: Build the image locally to confirm it actually builds and runs**

Run:
```bash
cd ~/data-science-portfolio/05-fraud-detection-api
docker build -t fraud-detection-api-local .
docker run -d -p 8001:8000 --name fraud-api-test fraud-detection-api-local
sleep 2
curl -s http://localhost:8001/health
docker stop fraud-api-test && docker rm fraud-api-test
```
Expected: image builds without error, `curl` returns `{"status":"ok"}`. If Docker isn't available in this environment, skip this step and note in Task 4's README that the Dockerfile is untested-in-container (still valid Python-level testing was done in Task 2 Step 6) — do not claim it was verified if it wasn't actually run.

- [ ] **Step 5: Commit**

```bash
cd ~/data-science-portfolio
git add 05-fraud-detection-api/Dockerfile 05-fraud-detection-api/.dockerignore 05-fraud-detection-api/render.yaml
git commit -m "Project 5: Docker + Render deployment config (not deployed)"
```

---

### Task 4: README and repo integration

**Files:**
- Create: `05-fraud-detection-api/README.md`
- Modify: `README.md` (repo root, row for project 5)
- Delete: `05-fraud-detection-api/.gitkeep`

**Interfaces:**
- Consumes: the real console output from Task 1 Step 5 (`model/metrics.json` values).
- Produces: nothing consumed elsewhere — terminal task for project 5.

- [ ] **Step 1: Write `README.md`**, replacing every bracketed value with the real number from `model/metrics.json` / Task 1's console output — no bracket may remain in the committed file

```markdown
# Fraud Detection Live API

A real RandomForest classifier trained on the real ULB Credit Card Fraud
dataset, deployed as a rate-limited FastAPI endpoint.

**Real dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
via Kaggle — 284,807 real anonymized European credit card transactions,
[N] real frauds ([X]% fraud rate). Not simulated.

## Real model performance (held-out test set)

Accuracy is not reported as the headline number — with ~99.8% of
transactions legitimate, a model that always predicts "legitimate" would
score ~99.8% accuracy while catching zero fraud. These are the metrics
that actually matter on imbalanced fraud data:

| Metric | Value |
|--------|-------|
| PR-AUC | [pr_auc] |
| Precision (fraud class) | [precision_fraud] |
| Recall (fraud class) | [recall_fraud] |
| F1 (fraud class) | [f1_fraud] |
| Test set size | [n_test] transactions, [n_test_fraud] real frauds |

**What this tells you**: [one sentence interpreting the real
precision/recall trade-off above — e.g. what fraction of real fraud this
model catches, and at what false-positive cost].

## API

- `GET /health` — liveness check.
- `GET /metrics` — the real training metrics above, as JSON.
- `POST /predict` — takes a transaction's `V1`-`V28` (PCA-anonymized) and
  `Amount` fields, returns `{"fraud_probability": float, "is_fraud": bool}`.
- Interactive docs (live demo): `<API_URL>/docs`

## Guardrails

- Rate-limited to 10 requests/minute/IP (`slowapi`, in-memory — no
  external service, no added cost as traffic grows).
- No paid API dependency — the model is a local scikit-learn artifact
  loaded from disk; nothing to leak, nothing billed per call.
- Deployed on Render's free tier, `autoDeploy: false` — going live is a
  deliberate, separate action, not automatic on push.
- Render's free tier sleeps after inactivity — the first request after a
  period of no traffic will be slow (cold start); this is expected.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-train.txt
python train_model.py               # trains on the real dataset (needs Kaggle API creds)
pip install -r requirements.txt
pip install pytest httpx
python -m pytest tests/ -v          # verify the API
uvicorn api.main:app --port 8000    # run it
```

Or with Docker: `docker build -t fraud-detection-api . && docker run -p 8000:8000 fraud-detection-api`

## Live demo

[Render URL — added after deployment approval and actual deploy]

## Notes

- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`) only for
  `train_model.py`; the deployed API itself needs no external
  credentials and makes no external calls.
- `model/fraud_model.joblib` is committed to this repo so the API can
  start without retraining.
- Every number above comes from `train_model.py`'s real run against the
  real dataset — see `model/metrics.json`.
```

- [ ] **Step 2: Update the root README table row for project 5**

In `~/data-science-portfolio/README.md`, replace the project 5 `_pending_` demo cell with the actual Render URL once deployed (or "local only, deploy pending approval" until then).

- [ ] **Step 3: Remove placeholder and commit**

```bash
cd ~/data-science-portfolio
git rm 05-fraud-detection-api/.gitkeep
git add 05-fraud-detection-api/README.md README.md
git commit -m "Project 5: README with real fraud model results"
```

- [ ] **Step 4: Final verification**

Run: `cd ~/data-science-portfolio/05-fraud-detection-api && source venv/bin/activate && python -m pytest tests/ -v`
Expected: all tests PASS. This is the completion gate for project 5 — deployment to Render happens only after Leanthel explicitly reviews this project and approves going live, as a separate follow-up action outside this plan.
