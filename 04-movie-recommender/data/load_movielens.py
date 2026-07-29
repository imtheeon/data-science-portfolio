"""Downloads the real MovieLens 100k dataset directly from grouplens.org
(no authentication needed). Real dataset: 100,000 real ratings from 943
real users on 1,682 real movies. Not simulated.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_DIR = Path(__file__).parent
ML_DIR = DATA_DIR / "ml-100k"
DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download() -> Path:
    if (ML_DIR / "u.data").exists():
        return ML_DIR
    DATA_DIR.mkdir(exist_ok=True)
    with urlopen(DOWNLOAD_URL) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DATA_DIR)
    if not (ML_DIR / "u.data").exists():
        raise FileNotFoundError(
            f"Expected ml-100k/u.data after extraction, found: {list(DATA_DIR.glob('*'))}"
        )
    return ML_DIR


def load_ratings() -> pd.DataFrame:
    ml_dir = download()
    return pd.read_csv(
        ml_dir / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )


def load_movies() -> pd.DataFrame:
    ml_dir = download()
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + [
        f"genre_{i}" for i in range(19)
    ]
    return pd.read_csv(ml_dir / "u.item", sep="|", encoding="ISO-8859-1", names=cols)


def load_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """MovieLens's official u1.base/u1.test 80/20 split — a standard,
    reproducible benchmark rather than an arbitrary random split."""
    ml_dir = download()
    cols = ["user_id", "item_id", "rating", "timestamp"]
    train = pd.read_csv(ml_dir / "u1.base", sep="\t", names=cols)
    test = pd.read_csv(ml_dir / "u1.test", sep="\t", names=cols)
    return train, test


if __name__ == "__main__":
    ratings = load_ratings()
    print(
        f"Loaded {len(ratings):,} real ratings from "
        f"{ratings['user_id'].nunique()} users on {ratings['item_id'].nunique()} movies."
    )
