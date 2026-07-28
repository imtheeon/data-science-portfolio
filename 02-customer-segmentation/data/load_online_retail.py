"""Downloads the real UK Online Retail transaction dataset from Kaggle.

Real transactional e-commerce data (invoice-level line items with customer
IDs), used for genuine RFM feature engineering — not simulated.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
KAGGLE_DATASET = "mashlyn/online-retail-ii-uci"


def download() -> Path:
    existing = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if existing:
        return existing[0]

    import kaggle

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
