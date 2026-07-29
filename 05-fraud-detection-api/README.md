# Fraud Detection Live API

A real RandomForest classifier trained on the real ULB Credit Card Fraud
dataset, packaged as a rate-limited FastAPI endpoint, configured to deploy
on Render's free tier (not yet deployed — see Live demo section).

**Real dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
via Kaggle — 284,807 real anonymized European credit card transactions,
492 real frauds (0.1727% fraud rate). Not simulated.

## Real model performance (held-out test set)

Accuracy is not reported as the headline number — with ~99.8% of
transactions legitimate, a model that always predicts "legitimate" would
score ~99.8% accuracy while catching zero fraud. These are the metrics
that actually matter on imbalanced fraud data:

| Metric | Value |
|--------|-------|
| PR-AUC | 0.8301 |
| Precision (fraud class) | 0.84 |
| Recall (fraud class) | 0.81 |
| F1 (fraud class) | 0.82 |
| Test set size | 56,962 transactions, 98 real frauds |

**What this tells you**: the model catches about 81% of real fraud in the
test set (79 of 98 fraudulent transactions), at a cost of some false
positives (recall 0.81 and precision 0.84 mean roughly 15 legitimate
transactions get flagged for every 79 frauds caught) — a deliberate
trade-off for imbalanced fraud data, not near-perfect detection.

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
- Configured to deploy on Render's free tier via `render.yaml` (repo root),
  `autoDeploy: false` — going live is a deliberate, separate action, not
  automatic on push. Not yet deployed.
- Once deployed, Render's free tier sleeps after inactivity — the first
  request after a period of no traffic will be slow (cold start); this is
  expected.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-train.txt
python train_model.py               # trains on the real dataset (needs Kaggle API creds)
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest tests/ -v          # verify the API
uvicorn api.main:app --port 8000    # run it
```

Or with Docker (run from inside `05-fraud-detection-api/`):
`docker build -t fraud-detection-api . && docker run -p 8000:8000 fraud-detection-api`

Note: this Dockerfile has not been build-tested in this environment (Docker
was unavailable); it follows standard patterns and Task 2's Python-level
tests cover the actual application logic, but treat the container build
itself as unverified until run for real.

## Live demo

Not yet deployed — local only, deploy pending Leanthel's review and
approval as a separate follow-up action.

## Notes

- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`) only for
  `train_model.py`; the deployed API itself needs no external
  credentials and makes no external calls.
- `model/fraud_model.joblib` is committed to this repo so the API can
  start without retraining.
- Every number above comes from `train_model.py`'s real run against the
  real dataset — see `model/metrics.json`.
