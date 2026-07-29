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
    import kaggle

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
