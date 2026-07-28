# Project 2: Customer Segmentation via Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `02-customer-segmentation/` in the `data-science-portfolio` repo: RFM feature engineering + K-means clustering on a real retail transaction dataset, with each real cluster translated into a concrete marketing action.

**Architecture:** A small pipeline of pure, independently-testable functions (`rfm.py` → `clustering.py` → `segment_actions.py`) driven by an analysis script that loads real data, runs the pipeline, saves charts, and prints the numbers that go into the README. A Jupyter notebook wraps the same pipeline for a narrative walkthrough.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest, Kaggle API, Jupyter.

## Global Constraints

- No fabricated data or metrics: every number in the README comes from code in this repo actually being run. (Spec: "Hard requirement")
- This is explicitly an unsupervised segmentation project, not a classification project. (Spec: Project 2 description)
- Nothing pushed to GitHub or published to GitHub Pages until Leanthel reviews and approves. (Spec: "Publishing gate")
- Working directory for all tasks: `~/data-science-portfolio/02-customer-segmentation/`.

---

### Task 1: Project setup and real dataset acquisition

**Files:**
- Create: `02-customer-segmentation/requirements.txt`
- Create: `02-customer-segmentation/data/load_online_retail.py`
- Create: `02-customer-segmentation/tests/__init__.py`

**Interfaces:**
- Consumes: Kaggle API credentials at `~/.kaggle/kaggle.json` (already configured).
- Produces: `data.load_online_retail.download() -> pathlib.Path`, used by Task 2.

- [ ] **Step 1: Create the project structure and venv**

```bash
cd ~/data-science-portfolio/02-customer-segmentation
mkdir -p data analysis tests charts notebooks
touch analysis/__init__.py tests/__init__.py
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Write `requirements.txt` and install**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
pytest==8.3.2
kaggle==1.6.17
jupyter==1.0.0
openpyxl==3.1.5
```

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Find the exact real dataset slug on Kaggle**

Run: `kaggle datasets list -s "online retail" --csv | head -20`
From the results, pick the real UK online-retail transaction dataset (columns should include something equivalent to `InvoiceNo`/`Invoice`, `InvoiceDate`, `CustomerID`/`Customer ID`, `Quantity`, `UnitPrice`, `Description` — the canonical UCI "Online Retail" / "Online Retail II" dataset). Record the exact `ref` (owner/dataset-slug) from the output — this is the value used in Step 4 below. Do not guess a slug without confirming it from this command's real output.

- [ ] **Step 4: Write the loader using the confirmed slug**

```python
# data/load_online_retail.py
"""Downloads the real UK Online Retail transaction dataset from Kaggle.

Real transactional e-commerce data (invoice-level line items with customer
IDs), used for genuine RFM feature engineering — not simulated.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

from pathlib import Path

import kaggle
import pandas as pd

DATA_DIR = Path(__file__).parent
KAGGLE_DATASET = "REPLACE_WITH_CONFIRMED_REF_FROM_STEP_3"  # e.g. "owner/online-retail-ii-uci"


def download() -> Path:
    existing = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if existing:
        return existing[0]

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)

    downloaded = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not downloaded:
        raise FileNotFoundError(f"No CSV/XLSX found in {DATA_DIR} after download.")
    return downloaded[0]


def load_raw() -> pd.DataFrame:
    path = download()
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="ISO-8859-1")


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} raw transaction rows, columns: {df.columns.tolist()}")
```

Before moving on, replace `KAGGLE_DATASET` with the real ref found in Step 3.

- [ ] **Step 5: Run it and confirm real data loads**

Run: `python -m data.load_online_retail`
Expected: prints a real row count (hundreds of thousands of rows) and the actual column names. Note the actual column names here — if they differ from `InvoiceNo`/`CustomerID`/`InvoiceDate`/`Quantity`/`UnitPrice` (e.g. `Invoice`/`Customer ID`), use the real names in Task 2.

- [ ] **Step 6: Commit**

```bash
cd ~/data-science-portfolio
git add 02-customer-segmentation/requirements.txt 02-customer-segmentation/data/load_online_retail.py 02-customer-segmentation/tests/__init__.py
git commit -m "Project 2: real Online Retail dataset loader"
```

---

### Task 2: RFM feature engineering

**Files:**
- Create: `02-customer-segmentation/analysis/rfm.py`
- Test: `02-customer-segmentation/tests/test_rfm.py`

**Interfaces:**
- Consumes: raw transaction DataFrame with (real, confirmed in Task 1) columns for invoice ID, customer ID, invoice date, quantity, unit price.
- Produces: `analysis.rfm.compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame` with columns `CustomerID`, `Recency`, `Frequency`, `Monetary`, used by Task 3 and Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rfm.py
import pandas as pd

from analysis.rfm import compute_rfm


def test_compute_rfm_basic():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "2", "3"],
            "CustomerID": [100, 100, 100, 200],
            "InvoiceDate": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-10", "2024-01-15"]
            ),
            "Quantity": [2, 1, 3, 5],
            "UnitPrice": [10.0, 5.0, 2.0, 4.0],
        }
    )
    snapshot = pd.Timestamp("2024-01-16")

    rfm = compute_rfm(df, snapshot_date=snapshot)

    cust100 = rfm[rfm["CustomerID"] == 100].iloc[0]
    cust200 = rfm[rfm["CustomerID"] == 200].iloc[0]

    assert cust100["Recency"] == 6          # snapshot - 2024-01-10
    assert cust100["Frequency"] == 2         # 2 distinct invoices
    assert cust100["Monetary"] == 2 * 10.0 + 1 * 5.0 + 3 * 2.0  # 31.0

    assert cust200["Recency"] == 1           # snapshot - 2024-01-15
    assert cust200["Frequency"] == 1
    assert cust200["Monetary"] == 5 * 4.0    # 20.0


def test_compute_rfm_excludes_cancelled_and_missing_customer():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "C2", "3"],  # "C..." = cancellation/return
            "CustomerID": [100, 100, None],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Quantity": [2, -1, 5],
            "UnitPrice": [10.0, 10.0, 4.0],
        }
    )
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2024-01-04"))
    assert len(rfm) == 1
    assert rfm.iloc[0]["CustomerID"] == 100
    assert rfm.iloc[0]["Frequency"] == 1
    assert rfm.iloc[0]["Monetary"] == 20.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_rfm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.rfm'`.

- [ ] **Step 3: Implement `analysis/rfm.py`**

```python
# analysis/rfm.py
"""RFM (Recency, Frequency, Monetary) feature engineering for customer
segmentation, computed from real invoice-level transaction data.

Cleaning rules
--------------
- Rows with a missing CustomerID are dropped (can't attribute to a segment).
- Cancelled/returned orders (InvoiceNo starting with "C") are excluded from
  Frequency and Monetary — they represent returns, not purchases.
"""

from __future__ import annotations

import pandas as pd


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return grouped.reset_index()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_rfm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 02-customer-segmentation/analysis/rfm.py 02-customer-segmentation/tests/test_rfm.py
git commit -m "Project 2: RFM feature engineering"
```

---

### Task 3: K-means clustering and k selection

**Files:**
- Create: `02-customer-segmentation/analysis/clustering.py`
- Test: `02-customer-segmentation/tests/test_clustering.py`

**Interfaces:**
- Consumes: RFM DataFrame with `Recency`, `Frequency`, `Monetary` columns (Task 2).
- Produces: `analysis.clustering.evaluate_k_range(rfm_df: pd.DataFrame, k_range: range) -> pd.DataFrame` (columns `k`, `inertia`, `silhouette`), `analysis.clustering.fit_kmeans(rfm_df: pd.DataFrame, k: int, random_state: int = 42) -> tuple[sklearn.cluster.KMeans, pandas.Series]` (returns the fitted model and a `Cluster` label Series aligned to `rfm_df`'s index), used by Task 4 and Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
import numpy as np
import pandas as pd

from analysis.clustering import evaluate_k_range, fit_kmeans


def _synthetic_rfm(seed: int = 0) -> pd.DataFrame:
    # Three well-separated blobs so clustering behavior is unambiguous to test.
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[5, 5, 5], scale=0.5, size=(30, 3))
    blob_b = rng.normal(loc=[50, 50, 50], scale=0.5, size=(30, 3))
    blob_c = rng.normal(loc=[100, 1, 100], scale=0.5, size=(30, 3))
    data = np.vstack([blob_a, blob_b, blob_c])
    return pd.DataFrame(data, columns=["Recency", "Frequency", "Monetary"])


def test_evaluate_k_range_returns_expected_shape():
    rfm = _synthetic_rfm()
    result = evaluate_k_range(rfm, range(2, 6))
    assert list(result["k"]) == [2, 3, 4, 5]
    assert (result["inertia"] > 0).all()
    assert result["silhouette"].between(-1, 1).all()


def test_fit_kmeans_recovers_three_blobs():
    rfm = _synthetic_rfm()
    model, labels = fit_kmeans(rfm, k=3)
    assert labels.nunique() == 3
    assert len(labels) == len(rfm)
    # Each of the three well-separated blobs should end up as its own cluster.
    counts = labels.value_counts()
    assert (counts == 30).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.clustering'`.

- [ ] **Step 3: Implement `analysis/clustering.py`**

```python
# analysis/clustering.py
"""K-means clustering over standardized RFM features, with k selected by
inertia (elbow) and silhouette score rather than picked arbitrarily."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


def _scaled(rfm_df: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(rfm_df[FEATURES])


def evaluate_k_range(rfm_df: pd.DataFrame, k_range: range, random_state: int = 42) -> pd.DataFrame:
    X = _scaled(rfm_df)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(rfm_df: pd.DataFrame, k: int, random_state: int = 42) -> tuple[KMeans, pd.Series]:
    X = _scaled(rfm_df)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, pd.Series(labels, index=rfm_df.index, name="Cluster")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 02-customer-segmentation/analysis/clustering.py 02-customer-segmentation/tests/test_clustering.py
git commit -m "Project 2: K-means clustering with k selection"
```

---

### Task 4: Segment profiling and marketing action mapping

**Files:**
- Create: `02-customer-segmentation/analysis/segment_actions.py`
- Test: `02-customer-segmentation/tests/test_segment_actions.py`

**Interfaces:**
- Consumes: RFM DataFrame with a `Cluster` column attached (Task 2 + Task 3 outputs joined).
- Produces: `analysis.segment_actions.profile_clusters(rfm_with_clusters: pd.DataFrame) -> pd.DataFrame` (one row per cluster: `Cluster`, `CustomerCount`, `PctCustomers`, `PctRevenue`, mean `Recency`/`Frequency`/`Monetary`), `analysis.segment_actions.label_segment(cluster_row: pd.Series, overall_medians: pd.Series) -> tuple[str, str]` (returns `(segment_name, marketing_action)`), used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segment_actions.py
import pandas as pd

from analysis.segment_actions import label_segment, profile_clusters


def test_profile_clusters_computes_real_shares():
    rfm = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Recency": [5, 5, 100, 100],
            "Frequency": [10, 10, 1, 1],
            "Monetary": [1000.0, 1000.0, 10.0, 10.0],
            "Cluster": [0, 0, 1, 1],
        }
    )
    profile = profile_clusters(rfm)
    cluster0 = profile[profile["Cluster"] == 0].iloc[0]
    assert cluster0["CustomerCount"] == 2
    assert cluster0["PctCustomers"] == 50.0
    assert cluster0["PctRevenue"] == pytest.approx(2000.0 / 2020.0 * 100, rel=1e-6)


def test_label_segment_champions_vs_at_risk():
    overall = pd.Series({"Recency": 50, "Frequency": 3, "Monetary": 200.0})

    champion_row = pd.Series({"Recency": 5, "Frequency": 12, "Monetary": 1500.0})
    name, action = label_segment(champion_row, overall)
    assert name == "Champions"
    assert "loyalty" in action.lower() or "reward" in action.lower()

    at_risk_row = pd.Series({"Recency": 200, "Frequency": 1, "Monetary": 20.0})
    name2, action2 = label_segment(at_risk_row, overall)
    assert name2 in {"At Risk", "Hibernating"}
    assert "win-back" in action2.lower() or "re-engage" in action2.lower()
```

Add `import pytest` to the top of the test file for `pytest.approx`.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_segment_actions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.segment_actions'`.

- [ ] **Step 3: Implement `analysis/segment_actions.py`**

```python
# analysis/segment_actions.py
"""Turns real cluster statistics into named segments with a concrete
marketing action — the mapping is a heuristic over each cluster's actual
computed Recency/Frequency/Monetary relative to the overall customer base,
not an arbitrary label."""

from __future__ import annotations

import pandas as pd


def profile_clusters(rfm_with_clusters: pd.DataFrame) -> pd.DataFrame:
    total_customers = len(rfm_with_clusters)
    total_revenue = rfm_with_clusters["Monetary"].sum()

    rows = []
    for cluster_id, group in rfm_with_clusters.groupby("Cluster"):
        rows.append(
            {
                "Cluster": cluster_id,
                "CustomerCount": len(group),
                "PctCustomers": len(group) / total_customers * 100,
                "PctRevenue": group["Monetary"].sum() / total_revenue * 100,
                "Recency": group["Recency"].mean(),
                "Frequency": group["Frequency"].mean(),
                "Monetary": group["Monetary"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("PctRevenue", ascending=False).reset_index(drop=True)


def label_segment(cluster_row: pd.Series, overall_medians: pd.Series) -> tuple[str, str]:
    recent = cluster_row["Recency"] <= overall_medians["Recency"]
    frequent = cluster_row["Frequency"] >= overall_medians["Frequency"]
    high_value = cluster_row["Monetary"] >= overall_medians["Monetary"]

    if recent and frequent and high_value:
        return "Champions", "Enroll in a loyalty/VIP rewards program to protect this high-value relationship."
    if recent and frequent:
        return "Loyal Customers", "Upsell/cross-sell campaigns — they buy often, grow their basket size."
    if recent and not frequent:
        return "New / Promising", "Onboarding nurture campaign to convert first purchase into a habit."
    if not recent and (frequent or high_value):
        return "At Risk", "Targeted win-back offer before they churn — they used to be valuable."
    return "Hibernating", "Low-cost re-engage email; deprioritize spend versus other segments."


def label_all_clusters(profile: pd.DataFrame) -> pd.DataFrame:
    overall_medians = profile[["Recency", "Frequency", "Monetary"]].median()
    labels = profile.apply(lambda row: label_segment(row, overall_medians), axis=1)
    profile = profile.copy()
    profile["Segment"] = [l[0] for l in labels]
    profile["MarketingAction"] = [l[1] for l in labels]
    return profile
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_segment_actions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 02-customer-segmentation/analysis/segment_actions.py 02-customer-segmentation/tests/test_segment_actions.py
git commit -m "Project 2: segment profiling and marketing action mapping"
```

---

### Task 5: Analysis script, charts, and notebook

**Files:**
- Create: `02-customer-segmentation/run_analysis.py`
- Create: `02-customer-segmentation/notebooks/segmentation.ipynb`

**Interfaces:**
- Consumes: `data.load_online_retail.load_raw` (Task 1), `analysis.rfm.compute_rfm` (Task 2), `analysis.clustering.{evaluate_k_range, fit_kmeans}` (Task 3), `analysis.segment_actions.{profile_clusters, label_all_clusters}` (Task 4).
- Produces: PNG charts in `charts/`, and printed real numbers used by Task 6's README. No other task consumes this directly.

- [ ] **Step 1: Write `run_analysis.py`**

```python
# run_analysis.py
"""End-to-end run: load real data -> RFM -> pick k -> cluster -> profile ->
label -> save charts. Prints the real numbers that go into the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from analysis.clustering import _scaled, evaluate_k_range, fit_kmeans
from analysis.rfm import compute_rfm
from analysis.segment_actions import label_all_clusters, profile_clusters
from data.load_online_retail import load_raw

CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw transaction rows.")

    rfm = compute_rfm(raw)
    print(f"Computed RFM for {len(rfm):,} real customers.")

    k_scores = evaluate_k_range(rfm, range(2, 9))
    print(k_scores.to_string(index=False))
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
    print(f"Selected k={best_k} (highest silhouette score).")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(k_scores["k"], k_scores["inertia"], marker="o")
    ax[0].set_title("Elbow: Inertia vs. k")
    ax[0].set_xlabel("k")
    ax[1].plot(k_scores["k"], k_scores["silhouette"], marker="o", color="darkorange")
    ax[1].set_title("Silhouette Score vs. k")
    ax[1].set_xlabel("k")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "k_selection.png", dpi=150)
    plt.close(fig)

    model, labels = fit_kmeans(rfm, k=best_k)
    rfm_clustered = rfm.copy()
    rfm_clustered["Cluster"] = labels

    profile = profile_clusters(rfm_clustered)
    profile = label_all_clusters(profile)
    print(profile.to_string(index=False))
    profile.to_csv(CHARTS_DIR.parent / "cluster_profile.csv", index=False)

    X_scaled = _scaled(rfm_clustered)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=rfm_clustered["Cluster"], cmap="tab10", alpha=0.6, s=15)
    ax2.set_title(f"Customer Segments (PCA projection, k={best_k})")
    legend = ax2.legend(*scatter.legend_elements(), title="Cluster")
    ax2.add_artist(legend)
    fig2.tight_layout()
    fig2.savefig(CHARTS_DIR / "pca_clusters.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.barh(profile["Segment"], profile["PctRevenue"])
    ax3.set_xlabel("% of total revenue")
    ax3.set_title("Revenue Share by Segment")
    fig3.tight_layout()
    fig3.savefig(CHARTS_DIR / "revenue_by_segment.png", dpi=150)
    plt.close(fig3)

    print("Charts saved to charts/. Cluster profile saved to cluster_profile.csv.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real data**

Run: `cd ~/data-science-portfolio/02-customer-segmentation && source venv/bin/activate && python run_analysis.py`
Expected: real printed row counts, k-selection table, chosen k, and the cluster profile table with real `PctCustomers`/`PctRevenue`/segment names/actions. Three PNGs appear in `charts/`, and `cluster_profile.csv` is created. Save this console output — it is the source of Task 6's README numbers.

- [ ] **Step 3: Build the notebook**

Create `notebooks/segmentation.ipynb` with markdown+code cells that narrate the same pipeline as `run_analysis.py` (import from `analysis.*` and `data.load_online_retail`, so there is exactly one implementation of the logic, not a duplicate copy): load data, compute RFM, show `k_scores` table + elbow/silhouette plots inline, fit final k-means, show the PCA scatter inline, show the labeled `profile` DataFrame, and a closing markdown cell stating the concrete marketing actions per segment. Run all cells top to bottom (`jupyter nbconvert --to notebook --execute --inplace notebooks/segmentation.ipynb`) so the committed notebook has real executed output, not empty cells.

- [ ] **Step 4: Commit**

```bash
cd ~/data-science-portfolio
git add 02-customer-segmentation/run_analysis.py 02-customer-segmentation/notebooks/segmentation.ipynb 02-customer-segmentation/charts 02-customer-segmentation/cluster_profile.csv
git commit -m "Project 2: analysis script, charts, and notebook"
```

---

### Task 6: README and repo integration

**Files:**
- Create: `02-customer-segmentation/README.md`
- Modify: `README.md` (repo root, row for project 2)
- Delete: `02-customer-segmentation/.gitkeep`

**Interfaces:**
- Consumes: the real console output and `cluster_profile.csv` from Task 5.
- Produces: nothing consumed elsewhere — terminal task for project 2.

- [ ] **Step 1: Write `README.md`** replacing every bracketed value with the real number from Task 5's output/`cluster_profile.csv` — do not leave any bracket in the committed file

```markdown
# Customer Segmentation via Clustering

RFM feature engineering + K-means clustering on real e-commerce transaction
data, with each segment translated into a concrete marketing action.

**Real dataset**: [Online Retail dataset](https://www.kaggle.com/datasets/REPLACE_WITH_ACTUAL_KAGGLE_REF) via Kaggle — [N] real transaction line items across [M] real customers. Not simulated.

## Segments found

k=[K] chosen by silhouette score ([score]) over k=2..8 — see `charts/k_selection.png`.

| Segment | % of customers | % of revenue | Avg. recency (days) | Avg. frequency | Avg. monetary | Action |
|---------|-----------------|----------------|----------------------|------------------|-----------------|--------|
| [Segment 1] | [X]% | [Y]% | [R] | [F] | $[M] | [Action] |
| ... | | | | | | |

**What this tells you**: [one sentence naming the highest-revenue segment
and the single highest-priority action from the real table above].

![PCA cluster projection](charts/pca_clusters.png)
![Revenue by segment](charts/revenue_by_segment.png)

## Method

- RFM (Recency, Frequency, Monetary) computed per customer from real
  invoice-level transactions (`analysis/rfm.py`), excluding cancellations
  and rows with no customer ID.
- Features standardized, K-means fit across k=2..8, k chosen by silhouette
  score (`analysis/clustering.py`).
- Each real cluster labeled against the overall customer base's median
  RFM values and mapped to a concrete marketing action
  (`analysis/segment_actions.py`).

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py       # real end-to-end run, saves charts/ and cluster_profile.csv
jupyter notebook notebooks/segmentation.ipynb   # narrative walkthrough
```

## Live demo

Static — see the charts above, or run the notebook locally. [GitHub Pages
link added after Leanthel enables Pages for this repo.]

## Notes

- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`); the raw
  dataset itself is not committed to this repo.
- Not a classification project — this is unsupervised segmentation,
  distinct from the classification-style projects already in the BI
  dashboard portfolio.
```

- [ ] **Step 2: Update the root README table row for project 2**

In `~/data-science-portfolio/README.md`, replace the project 2 `_pending_` demo cell with a link to `02-customer-segmentation/README.md` (or the GitHub Pages path once known).

- [ ] **Step 3: Remove placeholder and commit**

```bash
cd ~/data-science-portfolio
git rm 02-customer-segmentation/.gitkeep
git add 02-customer-segmentation/README.md README.md
git commit -m "Project 2: README with real segmentation results"
```

- [ ] **Step 4: Final verification**

Run: `cd ~/data-science-portfolio/02-customer-segmentation && source venv/bin/activate && python -m pytest -v`
Expected: all tests PASS. This is the completion gate for project 2.
