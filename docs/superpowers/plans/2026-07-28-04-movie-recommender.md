# Project 4: Movie Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `04-movie-recommender/` in the `data-science-portfolio` repo: item-based collaborative filtering on the real MovieLens 100k dataset, evaluated with RMSE and Precision@K on the official held-out test split, with concrete example recommendations.

**Architecture:** A data loader for the real MovieLens 100k dataset (direct download, no auth needed), a pure-function collaborative-filtering module (build user-item matrix → item-item cosine similarity → predict rating → recommend), a pure-function evaluation module (RMSE, Precision@K), and an analysis script that runs the real evaluation and prints real numbers. An optional Streamlit demo lets a user pick a real MovieLens user ID and see live recommendations.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn (cosine similarity only), pytest, Streamlit.

## Global Constraints

- No fabricated data or metrics: every number in the README comes from code in this repo actually being run. (Spec: "Hard requirement")
- Uses the real MovieLens 100k dataset from grouplens.org — no synthetic ratings.
- Nothing pushed to GitHub or deployed until Leanthel reviews and approves.
- Working directory for all tasks: `~/data-science-portfolio/04-movie-recommender/`.

---

### Task 1: Project setup and real dataset acquisition

**Files:**
- Create: `04-movie-recommender/requirements.txt`
- Create: `04-movie-recommender/data/load_movielens.py`
- Create: `04-movie-recommender/tests/__init__.py`

**Interfaces:**
- Consumes: network access to `files.grouplens.org` (no authentication required).
- Produces: `data.load_movielens.load_ratings() -> pandas.DataFrame` (columns `user_id, item_id, rating, timestamp`), `data.load_movielens.load_movies() -> pandas.DataFrame` (columns `item_id, title, ...`), `data.load_movielens.load_train_test_split() -> tuple[pandas.DataFrame, pandas.DataFrame]` (the official `u1.base`/`u1.test` 80/20 split, same columns as `load_ratings`) — used by Task 4 and Task 5.

- [ ] **Step 1: Create the project structure and venv**

```bash
cd ~/data-science-portfolio/04-movie-recommender
mkdir -p data analysis tests
touch analysis/__init__.py tests/__init__.py
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Write `requirements.txt` and install**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
pytest==8.3.2
streamlit==1.38.0
jupyter==1.0.0
```

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Write the loader**

```python
# data/load_movielens.py
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
```

- [ ] **Step 4: Run it and confirm real data loads**

Run: `python -m data.load_movielens`
Expected: prints `Loaded 100,000 real ratings from 943 users on 1682 movies.`

- [ ] **Step 5: Commit**

```bash
cd ~/data-science-portfolio
git add 04-movie-recommender/requirements.txt 04-movie-recommender/data/load_movielens.py 04-movie-recommender/tests/__init__.py
git commit -m "Project 4: real MovieLens 100k dataset loader"
```

---

### Task 2: Item-based collaborative filtering

**Files:**
- Create: `04-movie-recommender/analysis/collaborative_filtering.py`
- Test: `04-movie-recommender/tests/test_collaborative_filtering.py`

**Interfaces:**
- Consumes: ratings DataFrame with columns `user_id, item_id, rating` (Task 1).
- Produces: `analysis.collaborative_filtering.build_user_item_matrix(ratings: pandas.DataFrame) -> pandas.DataFrame` (rows=`user_id`, columns=`item_id`, 0 = unrated), `analysis.collaborative_filtering.compute_item_similarity(user_item_matrix: pandas.DataFrame) -> pandas.DataFrame` (square item x item cosine similarity), `analysis.collaborative_filtering.predict_rating(user_item_matrix, item_similarity, user_id: int, item_id: int, k: int = 20) -> float`, `analysis.collaborative_filtering.recommend_for_user(user_item_matrix, item_similarity, user_id: int, n: int = 5, k: int = 20) -> list[int]` — used by Task 3 (via `predict_rating`) and Task 4/5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_collaborative_filtering.py
import pandas as pd

from analysis.collaborative_filtering import (
    build_user_item_matrix,
    compute_item_similarity,
    predict_rating,
    recommend_for_user,
)


def _sample_ratings() -> pd.DataFrame:
    # 4 users, 4 items. Items 1 and 2 are co-rated very similarly by every
    # user (co-liked); item 3 diverges; item 4 is unrated by user 1.
    data = [
        (1, 1, 5), (1, 2, 5), (1, 3, 1),
        (2, 1, 4), (2, 2, 5), (2, 3, 1), (2, 4, 5),
        (3, 1, 1), (3, 2, 1), (3, 3, 5), (3, 4, 1),
        (4, 1, 5), (4, 2, 4), (4, 3, 2), (4, 4, 5),
    ]
    return pd.DataFrame(data, columns=["user_id", "item_id", "rating"])


def test_build_user_item_matrix_shape_and_values():
    matrix = build_user_item_matrix(_sample_ratings())
    assert matrix.shape == (4, 4)
    assert matrix.loc[1, 1] == 5
    assert matrix.loc[1, 4] == 0  # unrated -> filled with 0


def test_item_similarity_reflects_co_rating_pattern():
    matrix = build_user_item_matrix(_sample_ratings())
    sim = compute_item_similarity(matrix)
    assert sim.loc[1, 2] > sim.loc[1, 3]


def test_predict_rating_in_valid_range():
    matrix = build_user_item_matrix(_sample_ratings())
    sim = compute_item_similarity(matrix)
    predicted = predict_rating(matrix, sim, user_id=1, item_id=4, k=3)
    assert 0 <= predicted <= 5


def test_recommend_for_user_excludes_already_rated():
    matrix = build_user_item_matrix(_sample_ratings())
    sim = compute_item_similarity(matrix)
    recs = recommend_for_user(matrix, sim, user_id=1, n=3)
    rated = set(matrix.loc[1][matrix.loc[1] > 0].index)
    assert not set(recs) & rated
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_collaborative_filtering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.collaborative_filtering'`.

- [ ] **Step 3: Implement `analysis/collaborative_filtering.py`**

```python
# analysis/collaborative_filtering.py
"""Item-based collaborative filtering: recommend movies a user hasn't
rated yet based on their ratings for similar movies (cosine similarity
between items over the real user-rating vectors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(user_item_matrix.T.values)
    return pd.DataFrame(sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)


def predict_rating(
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    user_id: int,
    item_id: int,
    k: int = 20,
) -> float:
    """Weighted average of the user's ratings for the k most-similar items
    to `item_id` that they've actually rated: sum(sim * rating) / sum(|sim|)."""
    if item_id not in item_similarity.columns or user_id not in user_item_matrix.index:
        return float("nan")

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    if len(rated_items) == 0:
        return float("nan")

    sims = item_similarity.loc[item_id, rated_items]
    top_k = sims.abs().sort_values(ascending=False).head(k).index
    sims_top = sims.loc[top_k]
    ratings_top = user_ratings.loc[top_k]

    denom = sims_top.abs().sum()
    if denom == 0:
        return float("nan")
    return float((sims_top * ratings_top).sum() / denom)


def recommend_for_user(
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    user_id: int,
    n: int = 5,
    k: int = 20,
) -> list[int]:
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    predictions = {}
    for item_id in unrated_items:
        pred = predict_rating(user_item_matrix, item_similarity, user_id, item_id, k=k)
        if not np.isnan(pred):
            predictions[item_id] = pred

    ranked = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)
    return [item_id for item_id, _ in ranked[:n]]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_collaborative_filtering.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 04-movie-recommender/analysis/collaborative_filtering.py 04-movie-recommender/tests/test_collaborative_filtering.py
git commit -m "Project 4: item-based collaborative filtering"
```

---

### Task 3: Evaluation metrics

**Files:**
- Create: `04-movie-recommender/analysis/evaluation.py`
- Test: `04-movie-recommender/tests/test_evaluation.py`

**Interfaces:**
- Consumes: nothing beyond numpy.
- Produces: `analysis.evaluation.rmse(predictions: list[float], actuals: list[float]) -> float`, `analysis.evaluation.precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float` — used by Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation.py
import pytest

from analysis.evaluation import precision_at_k, rmse


def test_rmse_perfect_predictions():
    assert rmse([3, 4, 5], [3, 4, 5]) == 0.0


def test_rmse_known_value():
    assert rmse([4, 4], [3, 5]) == pytest.approx(1.0)


def test_precision_at_k_partial_hits():
    assert precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5) == pytest.approx(3 / 5)


def test_precision_at_k_no_hits():
    assert precision_at_k([10, 20], {1, 2, 3}, k=2) == 0.0


def test_precision_at_k_empty_recommendations():
    assert precision_at_k([], {1, 2}, k=5) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.evaluation'`.

- [ ] **Step 3: Implement `analysis/evaluation.py`**

```python
# analysis/evaluation.py
"""Standard recommender-system evaluation metrics: RMSE for rating
prediction accuracy, Precision@K for top-N recommendation quality."""

from __future__ import annotations

import numpy as np


def rmse(predictions: list[float], actuals: list[float]) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_evaluation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 04-movie-recommender/analysis/evaluation.py 04-movie-recommender/tests/test_evaluation.py
git commit -m "Project 4: RMSE and Precision@K evaluation"
```

---

### Task 4: Analysis script, notebook — real evaluation and example recommendations

**Files:**
- Create: `04-movie-recommender/run_analysis.py`
- Create: `04-movie-recommender/notebooks/recommender.ipynb`

**Interfaces:**
- Consumes: `data.load_movielens.{load_movies, load_train_test_split}` (Task 1), `analysis.collaborative_filtering.{build_user_item_matrix, compute_item_similarity, predict_rating, recommend_for_user}` (Task 2), `analysis.evaluation.{rmse, precision_at_k}` (Task 3).
- Produces: console output consumed by Task 6's README. No other task depends on this.

- [ ] **Step 1: Write `run_analysis.py`**

```python
# run_analysis.py
"""End-to-end run: load real MovieLens 100k data -> build item-based CF ->
evaluate RMSE and Precision@K on the official train/test split -> print
real example recommendations for a few users."""

from __future__ import annotations

import numpy as np

from analysis.collaborative_filtering import (
    build_user_item_matrix,
    compute_item_similarity,
    predict_rating,
    recommend_for_user,
)
from analysis.evaluation import precision_at_k, rmse
from data.load_movielens import load_movies, load_train_test_split

K_NEIGHBORS = 20
TOP_N = 5


def main() -> None:
    train, test = load_train_test_split()
    movies = load_movies().set_index("item_id")["title"]
    print(f"Train: {len(train):,} real ratings. Test: {len(test):,} real ratings.")

    matrix = build_user_item_matrix(train)
    similarity = compute_item_similarity(matrix)

    predictions, actuals = [], []
    for row in test.itertuples():
        if row.user_id not in matrix.index or row.item_id not in matrix.columns:
            continue
        pred = predict_rating(matrix, similarity, row.user_id, row.item_id, k=K_NEIGHBORS)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row.rating)

    score = rmse(predictions, actuals)
    print(f"RMSE on {len(predictions):,} real held-out test ratings: {score:.4f}")

    precisions = []
    sample_users = test["user_id"].unique()[:50]
    for user_id in sample_users:
        if user_id not in matrix.index:
            continue
        relevant = set(test[(test["user_id"] == user_id) & (test["rating"] >= 4)]["item_id"])
        if not relevant:
            continue
        recs = recommend_for_user(matrix, similarity, user_id, n=TOP_N, k=K_NEIGHBORS)
        precisions.append(precision_at_k(recs, relevant, TOP_N))
    mean_precision = float(np.mean(precisions)) if precisions else float("nan")
    print(
        f"Mean Precision@{TOP_N} over {len(precisions)} real users with "
        f"relevant held-out items: {mean_precision:.4f}"
    )

    print("\nExample recommendations:")
    for user_id in sample_users[:3]:
        if user_id not in matrix.index:
            continue
        recs = recommend_for_user(matrix, similarity, user_id, n=TOP_N, k=K_NEIGHBORS)
        titles = [movies.get(i, f"item {i}") for i in recs]
        print(f"  User {user_id}: {titles}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real data**

Run: `cd ~/data-science-portfolio/04-movie-recommender && source venv/bin/activate && python run_analysis.py`
Expected: real printed RMSE, mean Precision@5, and 3 real example recommendation lists with actual movie titles. This may take a few minutes on the full ~20,000-row test split — let it finish. Save this console output; it is the source of Task 6's README numbers.

- [ ] **Step 3: Build the notebook**

Create `notebooks/recommender.ipynb` with markdown+code cells that narrate the same pipeline as `run_analysis.py` (import from `analysis.*` and `data.load_movielens`, so there is exactly one implementation of the logic, not a duplicate copy): load the real train/test split, build the user-item matrix, compute item similarity, show the RMSE and Precision@5 evaluation inline, and a closing cell printing example recommendations (with real movie titles) for 3 real user IDs. Run all cells top to bottom (`jupyter nbconvert --to notebook --execute --inplace notebooks/recommender.ipynb`) so the committed notebook has real executed output.

- [ ] **Step 4: Commit**

```bash
cd ~/data-science-portfolio
git add 04-movie-recommender/run_analysis.py 04-movie-recommender/notebooks/recommender.ipynb
git commit -m "Project 4: real MovieLens evaluation run and notebook"
```

---

### Task 5: Optional Streamlit demo

**Files:**
- Create: `04-movie-recommender/app.py`

**Interfaces:**
- Consumes: `data.load_movielens.{load_ratings, load_movies}` (Task 1), `analysis.collaborative_filtering.{build_user_item_matrix, compute_item_similarity, recommend_for_user}` (Task 2).
- Produces: a runnable Streamlit app; no other task depends on it.

- [ ] **Step 1: Write `app.py`**

```python
# app.py
"""Streamlit demo: pick a real MovieLens user, see live recommendations
computed by the same item-based CF logic used in run_analysis.py."""

from __future__ import annotations

import streamlit as st

from analysis.collaborative_filtering import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_for_user,
)
from data.load_movielens import load_movies, load_ratings

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommender (Item-Based Collaborative Filtering)")
st.caption(
    "Real MovieLens 100k ratings — recommendations computed live from "
    "real rating patterns, not hardcoded."
)


@st.cache_data
def _load():
    ratings = load_ratings()
    movies = load_movies().set_index("item_id")["title"]
    matrix = build_user_item_matrix(ratings)
    similarity = compute_item_similarity(matrix)
    return matrix, similarity, movies


matrix, similarity, movies = _load()

user_id = st.selectbox("Pick a real MovieLens user ID", sorted(matrix.index.tolist()))
n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend"):
    recs = recommend_for_user(matrix, similarity, user_id, n=n)
    st.write(f"**Top {n} recommendations for user {user_id}:**")
    for item_id in recs:
        st.write(f"- {movies.get(item_id, f'item {item_id}')}")

    st.write("**Movies this user already rated highly (>=4):**")
    already = matrix.loc[user_id]
    liked = already[already >= 4].index[:10]
    for item_id in liked:
        st.write(f"- {movies.get(item_id, f'item {item_id}')}")
```

- [ ] **Step 2: Run the app locally and verify it responds**

Run: `streamlit run app.py --server.headless true &` then `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501`; kill the background process afterward.
Expected: HTTP 200, no startup exceptions.

- [ ] **Step 3: Commit**

```bash
cd ~/data-science-portfolio
git add 04-movie-recommender/app.py
git commit -m "Project 4: Streamlit demo"
```

---

### Task 6: README and repo integration

**Files:**
- Create: `04-movie-recommender/README.md`
- Modify: `README.md` (repo root, row for project 4)
- Delete: `04-movie-recommender/.gitkeep`

**Interfaces:**
- Consumes: the real console output from Task 4 Step 2.
- Produces: nothing consumed elsewhere — terminal task for project 4.

- [ ] **Step 1: Write `README.md`**, replacing every bracketed value with the real number from Task 4's output — no bracket may remain in the committed file

```markdown
# Movie Recommender

Item-based collaborative filtering on the real MovieLens 100k dataset,
evaluated on the official held-out test split.

**Real dataset**: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
— 100,000 real ratings from 943 real users on 1,682 real movies. Not
simulated.

## Results

Evaluated on the official `u1.base`/`u1.test` 80/20 split:

- **RMSE**: [score] on [N] real held-out ratings
- **Precision@5**: [precision] averaged over [N] real users with relevant held-out items

**What this tells you**: [one sentence interpreting the real RMSE/Precision@5
numbers above — e.g. how they compare to the 1-5 rating scale / a random
baseline].

## Example recommendations (real output)

| User | Top 5 recommended movies |
|------|---------------------------|
| [user_id] | [titles] |
| [user_id] | [titles] |
| [user_id] | [titles] |

## Method

- Item-based collaborative filtering: cosine similarity between movies
  over real user-rating vectors (`analysis/collaborative_filtering.py`).
- Predicted rating = similarity-weighted average of the user's real
  ratings for the k=20 most similar movies they've actually rated.
- Evaluated with RMSE (rating-prediction accuracy) and Precision@5
  (top-N recommendation quality) on the official held-out test split.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py     # real evaluation run
streamlit run app.py       # interactive demo
```

## Live demo

[Streamlit Community Cloud link — added after deployment approval]

## Notes

- Dataset downloads automatically from grouplens.org on first run (no
  account/auth needed); not committed to this repo.
- Every number above comes from `run_analysis.py` — see that file for
  the exact computation.
```

- [ ] **Step 2: Update the root README table row for project 4**

In `~/data-science-portfolio/README.md`, replace the project 4 `_pending_` demo cell with the actual Streamlit Cloud link once deployed (or "local only" until then).

- [ ] **Step 3: Remove placeholder and commit**

```bash
cd ~/data-science-portfolio
git rm 04-movie-recommender/.gitkeep
git add 04-movie-recommender/README.md README.md
git commit -m "Project 4: README with real MovieLens results"
```

- [ ] **Step 4: Final verification**

Run: `cd ~/data-science-portfolio/04-movie-recommender && source venv/bin/activate && python -m pytest -v`
Expected: all tests PASS. This is the completion gate for project 4.
