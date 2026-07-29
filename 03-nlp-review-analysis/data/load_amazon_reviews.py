"""Downloads the real Amazon Fine Food Reviews dataset from Kaggle and
returns a reproducible random sample.

Real dataset (not simulated): ~568k real Amazon reviews with 1-5 star
ratings and review text. The full dataset is large, so this project
analyzes a fixed, reproducible random sample - stated honestly in the
README rather than implying the full dataset was used.

Confirmed via `kaggle datasets list -s "amazon fine food reviews" --csv`
(Task 1, Step 3): ref `snap/amazon-fine-food-reviews` is the real,
well-known Amazon Fine Food Reviews dataset (242MB, ~264k downloads,
2454 votes).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"  # confirmed in Task 1 Step 3
KAGGLE_CSV_NAME = "Reviews.csv"  # the known real filename for this dataset
SAMPLE_SIZE = 20_000
SAMPLE_SEED = 42


def _pick_csv() -> Path | None:
    """Prefer the known real dataset filename over an arbitrary glob match,
    since DATA_DIR can also contain other loose CSVs."""
    preferred = DATA_DIR / KAGGLE_CSV_NAME
    if preferred.exists():
        return preferred
    existing = list(DATA_DIR.glob("*.csv"))
    return existing[0] if existing else None


def download() -> Path:
    existing = _pick_csv()
    if existing:
        return existing

    # Imported locally (not at module scope) so that importing this module
    # does not require kaggle credentials or network access - keeps offline
    # unit tests fast and able to import the module without side effects.
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    downloaded = _pick_csv()
    if not downloaded:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR} after download.")
    return downloaded


def load_sample(n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, usecols=["Score", "Text"])
    df = df.dropna(subset=["Score", "Text"])
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = load_sample()
    print(f"Loaded a real sample of {len(df):,} Amazon Fine Food reviews.")
    print(df["Score"].value_counts().sort_index())
