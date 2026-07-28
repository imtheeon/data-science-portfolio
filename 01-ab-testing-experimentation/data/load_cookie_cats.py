# data/load_cookie_cats.py
"""Downloads the real Cookie Cats mobile-game A/B test dataset from Kaggle.

Dataset: mursideyarkin/mobile-games-ab-testing-cookie-cats — real player
records (userid, experiment arm, rounds played, 1-day and 7-day retention).
This is genuine, not simulated, data; nothing in this file invents numbers,
it only fetches and locates the real CSV.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "cookie_cats.csv"
KAGGLE_DATASET = "mursideyarkin/mobile-games-ab-testing-cookie-cats"


def download() -> Path:
    if CSV_PATH.exists():
        return CSV_PATH

    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
    )

    if not CSV_PATH.exists():
        found = list(DATA_DIR.glob("*.csv"))
        if len(found) == 1:
            found[0].rename(CSV_PATH)
        else:
            raise FileNotFoundError(
                f"Expected cookie_cats.csv after Kaggle download, found: {found}"
            )
    return CSV_PATH


if __name__ == "__main__":
    path = download()
    print(f"Cookie Cats dataset ready at {path}")
