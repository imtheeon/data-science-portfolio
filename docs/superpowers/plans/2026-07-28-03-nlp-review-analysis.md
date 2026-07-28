# Project 3: NLP Sentiment & Topic Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `03-nlp-review-analysis/` in the `data-science-portfolio` repo: sentiment classification and topic extraction on a real public review dataset, using classic/local NLP (VADER + NMF topic modeling) — no paid API, fully reproducible.

**Architecture:** Two independent, testable modules (`sentiment.py`, `topics.py`) driven by an analysis script that loads a real, honestly-sized sample of Amazon Fine Food Reviews, scores sentiment, checks it against the real star ratings, extracts topics, and saves charts + model artifacts. A small optional Streamlit demo reuses the saved artifacts for live inference on user-typed text.

**Tech Stack:** Python 3.12, pandas, nltk (VADER), scikit-learn (TF-IDF + NMF), matplotlib, joblib, pytest, Streamlit, Kaggle API.

## Global Constraints

- No fabricated data or metrics: every number in the README comes from code in this repo actually being run. (Spec: "Hard requirement")
- Classic/local NLP approach chosen over the Claude API specifically for zero cost and full reproducibility — decided during design review. (Spec: Project 3)
- The dataset is large; this project analyzes a reproducible random sample, and the README states the sample size honestly rather than implying the full dataset was used.
- Nothing pushed to GitHub or deployed until Leanthel reviews and approves.
- Working directory for all tasks: `~/data-science-portfolio/03-nlp-review-analysis/`.

---

### Task 1: Project setup and real dataset acquisition

**Files:**
- Create: `03-nlp-review-analysis/requirements.txt`
- Create: `03-nlp-review-analysis/data/load_amazon_reviews.py`
- Create: `03-nlp-review-analysis/tests/__init__.py`

**Interfaces:**
- Consumes: Kaggle API credentials at `~/.kaggle/kaggle.json`.
- Produces: `data.load_amazon_reviews.load_sample(n: int = 20000, seed: int = 42) -> pandas.DataFrame` with columns `Score` (int, 1-5) and `Text` (str), used by Task 4.

- [ ] **Step 1: Create the project structure and venv**

```bash
cd ~/data-science-portfolio/03-nlp-review-analysis
mkdir -p data analysis tests charts model
touch analysis/__init__.py tests/__init__.py
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Write `requirements.txt` and install**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
nltk==3.8.1
matplotlib==3.9.1
joblib==1.4.2
pytest==8.3.2
kaggle==1.6.17
streamlit==1.38.0
jupyter==1.0.0
```

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

- [ ] **Step 3: Confirm the exact Kaggle dataset ref**

Run: `kaggle datasets list -s "amazon fine food reviews" --csv | head -10`
Confirm the ref `snap/amazon-fine-food-reviews` appears (this is the well-known real Amazon Fine Food Reviews dataset — ~568k real reviews with `Score` 1-5 and `Text` columns). If the exact ref differs in the real output, use the actual ref found.

- [ ] **Step 4: Write the loader**

```python
# data/load_amazon_reviews.py
"""Downloads the real Amazon Fine Food Reviews dataset from Kaggle and
returns a reproducible random sample.

Real dataset (not simulated): ~568k real Amazon reviews with 1-5 star
ratings and review text. The full dataset is large, so this project
analyzes a fixed, reproducible random sample — stated honestly in the
README rather than implying the full dataset was used.
"""

from __future__ import annotations

from pathlib import Path

import kaggle
import pandas as pd

DATA_DIR = Path(__file__).parent
KAGGLE_DATASET = "snap/amazon-fine-food-reviews"  # confirmed in Task 1 Step 3
SAMPLE_SIZE = 20_000
SAMPLE_SEED = 42


def download() -> Path:
    existing = list(DATA_DIR.glob("*.csv"))
    if existing:
        return existing[0]
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    downloaded = list(DATA_DIR.glob("*.csv"))
    if not downloaded:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR} after download.")
    return downloaded[0]


def load_sample(n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, usecols=["Score", "Text"])
    df = df.dropna(subset=["Score", "Text"])
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = load_sample()
    print(f"Loaded a real sample of {len(df):,} Amazon Fine Food reviews.")
    print(df["Score"].value_counts().sort_index())
```

- [ ] **Step 5: Run it and confirm real data loads**

Run: `python -m data.load_amazon_reviews`
Expected: prints "Loaded a real sample of 20,000 Amazon Fine Food reviews." and a real distribution of star ratings 1-5.

- [ ] **Step 6: Commit**

```bash
cd ~/data-science-portfolio
git add 03-nlp-review-analysis/requirements.txt 03-nlp-review-analysis/data/load_amazon_reviews.py 03-nlp-review-analysis/tests/__init__.py
git commit -m "Project 3: real Amazon Fine Food Reviews dataset loader"
```

---

### Task 2: Sentiment scoring

**Files:**
- Create: `03-nlp-review-analysis/analysis/sentiment.py`
- Test: `03-nlp-review-analysis/tests/test_sentiment.py`

**Interfaces:**
- Consumes: nothing beyond `nltk`'s downloaded VADER lexicon (Task 1).
- Produces: `analysis.sentiment.score_sentiment(texts: list[str]) -> pandas.DataFrame` (columns `compound: float`, `predicted_label: str` in `{"positive","neutral","negative"}`), `analysis.sentiment.rating_to_label(rating: int) -> str`, `analysis.sentiment.evaluate_agreement(predicted_labels: pandas.Series, true_labels: pandas.Series) -> dict` (keys `accuracy`, `n`, `n_correct`) — all used by Task 4 and Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sentiment.py
import pandas as pd
import pytest

from analysis.sentiment import evaluate_agreement, rating_to_label, score_sentiment


def test_score_sentiment_obvious_cases():
    df = score_sentiment([
        "I love this, it's amazing and wonderful!",
        "This is terrible, I hate it, worst ever.",
    ])
    assert df.iloc[0]["predicted_label"] == "positive"
    assert df.iloc[1]["predicted_label"] == "negative"
    assert df.iloc[0]["compound"] > 0
    assert df.iloc[1]["compound"] < 0


def test_rating_to_label():
    assert rating_to_label(5) == "positive"
    assert rating_to_label(4) == "positive"
    assert rating_to_label(1) == "negative"
    assert rating_to_label(2) == "negative"
    assert rating_to_label(3) == "neutral"


def test_evaluate_agreement():
    predicted = pd.Series(["positive", "negative", "positive"])
    true = pd.Series(["positive", "negative", "negative"])
    result = evaluate_agreement(predicted, true)
    assert result["n"] == 3
    assert result["n_correct"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.sentiment'`.

- [ ] **Step 3: Implement `analysis/sentiment.py`**

```python
# analysis/sentiment.py
"""VADER-based sentiment scoring (lexicon/rule-based, no paid API, fully
local and reproducible), plus helpers to compare predictions against real
star ratings."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_sentiment(texts: list[str]) -> pd.DataFrame:
    analyzer = _get_analyzer()
    rows = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        rows.append({"compound": compound, "predicted_label": label})
    return pd.DataFrame(rows)


def rating_to_label(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def evaluate_agreement(predicted_labels: pd.Series, true_labels: pd.Series) -> dict:
    matches = predicted_labels.reset_index(drop=True) == true_labels.reset_index(drop=True)
    return {
        "accuracy": float(matches.mean()),
        "n": int(len(predicted_labels)),
        "n_correct": int(matches.sum()),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sentiment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 03-nlp-review-analysis/analysis/sentiment.py 03-nlp-review-analysis/tests/test_sentiment.py
git commit -m "Project 3: VADER sentiment scoring"
```

---

### Task 3: Topic modeling

**Files:**
- Create: `03-nlp-review-analysis/analysis/topics.py`
- Test: `03-nlp-review-analysis/tests/test_topics.py`

**Interfaces:**
- Consumes: nothing beyond scikit-learn.
- Produces: `analysis.topics.fit_topic_model(texts: list[str], n_topics: int = 8, n_top_words: int = 10, random_state: int = 42) -> tuple[sklearn.decomposition.NMF, sklearn.feature_extraction.text.TfidfVectorizer, list[list[str]], numpy.ndarray]` (returns `model, vectorizer, topic_words, W` where `W` is the document-topic matrix), `analysis.topics.dominant_topics(W: numpy.ndarray) -> numpy.ndarray` — used by Task 4 and Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topics.py
from analysis.topics import dominant_topics, fit_topic_model


def test_fit_topic_model_separates_distinct_themes():
    dog_docs = ["dog puppy bark leash walk dog park dog treat"] * 10
    pizza_docs = ["pizza cheese pepperoni oven slice pizza box pizza delivery"] * 10
    texts = dog_docs + pizza_docs

    model, vectorizer, topic_words, W = fit_topic_model(texts, n_topics=2, n_top_words=5)

    assert len(topic_words) == 2
    all_words = {w for words in topic_words for w in words}
    assert "dog" in all_words
    assert "pizza" in all_words

    dominant = dominant_topics(W)
    dog_topics = set(dominant[:10])
    pizza_topics = set(dominant[10:])
    assert len(dog_topics) == 1
    assert len(pizza_topics) == 1
    assert dog_topics != pizza_topics
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_topics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.topics'`.

- [ ] **Step 3: Implement `analysis/topics.py`**

```python
# analysis/topics.py
"""Topic extraction via TF-IDF + NMF (Non-negative Matrix Factorization) —
a standard, fully local/free approach to unsupervised topic modeling."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_topic_model(
    texts: list[str],
    n_topics: int = 8,
    n_top_words: int = 10,
    random_state: int = 42,
):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)

    model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic in model.components_:
        top_indices = topic.argsort()[::-1][:n_top_words]
        topic_words.append([feature_names[i] for i in top_indices])

    return model, vectorizer, topic_words, W


def dominant_topics(W: np.ndarray) -> np.ndarray:
    return np.argmax(W, axis=1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_topics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 03-nlp-review-analysis/analysis/topics.py 03-nlp-review-analysis/tests/test_topics.py
git commit -m "Project 3: NMF topic modeling"
```

---

### Task 4: Analysis script, notebook — real run, charts, saved model artifacts

**Files:**
- Create: `03-nlp-review-analysis/run_analysis.py`
- Create: `03-nlp-review-analysis/notebooks/review_analysis.ipynb`

**Interfaces:**
- Consumes: `data.load_amazon_reviews.load_sample` (Task 1), `analysis.sentiment.{score_sentiment, rating_to_label, evaluate_agreement}` (Task 2), `analysis.topics.{fit_topic_model, dominant_topics}` (Task 3).
- Produces: `charts/sentiment_vs_rating.png`, `charts/topics.png`, `model/{vectorizer.joblib, nmf_model.joblib, topic_words.joblib}` (consumed by Task 5's Streamlit app), `analyzed_sample.csv`, and console output consumed by Task 6's README.

- [ ] **Step 1: Write `run_analysis.py`**

```python
# run_analysis.py
"""End-to-end run: load a real Amazon review sample -> VADER sentiment ->
compare against real star ratings -> NMF topic model -> save charts and
model artifacts for the Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from analysis.sentiment import evaluate_agreement, rating_to_label, score_sentiment
from analysis.topics import dominant_topics, fit_topic_model
from data.load_amazon_reviews import load_sample

CHARTS_DIR = Path(__file__).parent / "charts"
MODEL_DIR = Path(__file__).parent / "model"
CHARTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

N_TOPICS = 8


def main() -> None:
    df = load_sample()
    print(f"Analyzing a real sample of {len(df):,} Amazon Fine Food reviews.")

    sentiment_df = score_sentiment(df["Text"].tolist())
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    df["true_label"] = df["Score"].apply(rating_to_label)

    agreement = evaluate_agreement(df["predicted_label"], df["true_label"])
    print(
        f"VADER vs. star-rating agreement: {agreement['accuracy']:.4f} "
        f"({agreement['n_correct']}/{agreement['n']})"
    )

    crosstab = pd.crosstab(df["true_label"], df["predicted_label"])
    print(crosstab)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(crosstab.values, cmap="Blues")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index)
    ax.set_xlabel("VADER predicted label")
    ax.set_ylabel("Star-rating-derived label")
    ax.set_title(f"Sentiment vs. Rating Agreement (accuracy={agreement['accuracy']:.2%})")
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax.text(j, i, crosstab.values[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sentiment_vs_rating.png", dpi=150)
    plt.close(fig)

    model, vectorizer, topic_words, W = fit_topic_model(df["Text"].tolist(), n_topics=N_TOPICS)
    df["topic"] = dominant_topics(W)

    for i, words in enumerate(topic_words):
        print(f"Topic {i}: {', '.join(words)}")

    topic_sizes = df["topic"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    labels = [f"Topic {i}: {', '.join(topic_words[i][:3])}" for i in topic_sizes.index]
    ax2.barh(labels, topic_sizes.values)
    ax2.set_xlabel("Number of reviews")
    ax2.set_title("Review Volume by Topic")
    fig2.tight_layout()
    fig2.savefig(CHARTS_DIR / "topics.png", dpi=150)
    plt.close(fig2)

    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "nmf_model.joblib")
    joblib.dump(topic_words, MODEL_DIR / "topic_words.joblib")

    df.to_csv(CHARTS_DIR.parent / "analyzed_sample.csv", index=False)
    print("Charts saved to charts/, model artifacts saved to model/, full sample saved to analyzed_sample.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real sample**

Run: `cd ~/data-science-portfolio/03-nlp-review-analysis && source venv/bin/activate && python run_analysis.py`
Expected: real printed accuracy, crosstab, and 8 real topic word lists. `charts/sentiment_vs_rating.png`, `charts/topics.png`, `model/*.joblib`, and `analyzed_sample.csv` all exist afterward. Save this console output — it is the source of Task 6's README numbers.

- [ ] **Step 3: Build the notebook**

Create `notebooks/review_analysis.ipynb` with markdown+code cells that narrate the same pipeline as `run_analysis.py` (import from `analysis.*` and `data.load_amazon_reviews`, so there is exactly one implementation of the logic): load the real sample, score sentiment, show the crosstab/agreement inline, fit the topic model, show the topic word table and volume-by-topic chart inline, and a closing markdown cell stating the real headline accuracy and the most common topic. Run all cells top to bottom (`jupyter nbconvert --to notebook --execute --inplace notebooks/review_analysis.ipynb`) so the committed notebook has real executed output.

- [ ] **Step 4: Commit**

```bash
cd ~/data-science-portfolio
git add 03-nlp-review-analysis/run_analysis.py 03-nlp-review-analysis/notebooks/review_analysis.ipynb 03-nlp-review-analysis/charts 03-nlp-review-analysis/model 03-nlp-review-analysis/analyzed_sample.csv
git commit -m "Project 3: real sentiment/topic analysis run and notebook"
```

---

### Task 5: Optional Streamlit demo

**Files:**
- Create: `03-nlp-review-analysis/app.py`

**Interfaces:**
- Consumes: `model/{vectorizer.joblib, nmf_model.joblib, topic_words.joblib}` (Task 4), `analysis.sentiment.score_sentiment` (Task 2).
- Produces: a runnable Streamlit app; no other task depends on it.

- [ ] **Step 1: Write `app.py`**

```python
# app.py
"""Streamlit demo: type a review, see its predicted sentiment and nearest
topic — both computed live by the models trained in run_analysis.py."""

from __future__ import annotations

import joblib
import streamlit as st

from analysis.sentiment import score_sentiment

MODEL_DIR = "model"

st.set_page_config(page_title="NLP Review Analysis", layout="centered")
st.title("NLP Sentiment & Topic Analysis")
st.caption(
    "Real Amazon review data + classic/local NLP (VADER sentiment, NMF "
    "topic modeling) — no external API calls, fully local inference."
)

vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.joblib")
nmf_model = joblib.load(f"{MODEL_DIR}/nmf_model.joblib")
topic_words = joblib.load(f"{MODEL_DIR}/topic_words.joblib")

text = st.text_area(
    "Review text", "This product was amazing, fast shipping and great taste!"
)

if st.button("Analyze"):
    sentiment_df = score_sentiment([text])
    row = sentiment_df.iloc[0]
    st.metric("Predicted sentiment", row["predicted_label"], f"compound={row['compound']:.3f}")

    X = vectorizer.transform([text])
    topic_dist = nmf_model.transform(X)[0]
    top_topic = int(topic_dist.argmax())
    st.write(f"**Nearest topic**: Topic {top_topic} — {', '.join(topic_words[top_topic])}")
```

- [ ] **Step 2: Run the app locally and verify it responds**

Run: `streamlit run app.py --server.headless true &` then `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501`; kill the background process afterward.
Expected: HTTP 200, no startup exceptions.

- [ ] **Step 3: Commit**

```bash
cd ~/data-science-portfolio
git add 03-nlp-review-analysis/app.py
git commit -m "Project 3: Streamlit demo"
```

---

### Task 6: README and repo integration

**Files:**
- Create: `03-nlp-review-analysis/README.md`
- Modify: `README.md` (repo root, row for project 3)
- Delete: `03-nlp-review-analysis/.gitkeep`

**Interfaces:**
- Consumes: the real console output from Task 4 Step 2.
- Produces: nothing consumed elsewhere — terminal task for project 3.

- [ ] **Step 1: Write `README.md`**, replacing every bracketed value with the real number from Task 4's output — no bracket may remain in the committed file

```markdown
# NLP Sentiment & Topic Analysis

Sentiment classification and topic extraction on real Amazon product
reviews — classic/local NLP (VADER + NMF), no external API calls.

**Real dataset**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
via Kaggle. Analyzed on a reproducible random sample of [N] reviews (seed=42) out
of the full ~568k — stated honestly, not the full dataset.

## Sentiment: does VADER agree with the star rating?

Ratings 4-5 -> "positive", 3 -> "neutral", 1-2 -> "negative"; VADER's
compound score thresholded the same way.

**Agreement with real star ratings: [accuracy]%** ([n_correct]/[n])

![Sentiment vs. rating confusion matrix](charts/sentiment_vs_rating.png)

**What this tells you**: [one sentence — e.g. where VADER agrees strongly
vs. where it struggles, based on the real crosstab above].

## Topics found (NMF, 8 topics)

| Topic | Top words |
|-------|-----------|
| 0 | [words] |
| ... | |

![Review volume by topic](charts/topics.png)

## Method

- Sentiment: VADER (`nltk.sentiment.SentimentIntensityAnalyzer`), a
  lexicon/rule-based model — free, local, no training data needed
  (`analysis/sentiment.py`).
- Topics: TF-IDF + NMF over the review text (`analysis/topics.py`).
- Chosen over the Claude API specifically to keep this project free to
  run and fully reproducible by anyone who clones the repo.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
python run_analysis.py     # real end-to-end run
streamlit run app.py       # interactive demo
```

## Live demo

[Streamlit Community Cloud link — added after deployment approval]

## Notes

- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`); the raw
  dataset is not committed to this repo.
- Everything above is generated by `run_analysis.py` — see that file for
  the exact computation.
```

- [ ] **Step 2: Update the root README table row for project 3**

In `~/data-science-portfolio/README.md`, replace the project 3 `_pending_` demo cell with the actual Streamlit Cloud link once deployed (or "local only" until then).

- [ ] **Step 3: Remove placeholder and commit**

```bash
cd ~/data-science-portfolio
git rm 03-nlp-review-analysis/.gitkeep
git add 03-nlp-review-analysis/README.md README.md
git commit -m "Project 3: README with real sentiment/topic results"
```

- [ ] **Step 4: Final verification**

Run: `cd ~/data-science-portfolio/03-nlp-review-analysis && source venv/bin/activate && python -m pytest -v`
Expected: all tests PASS. This is the completion gate for project 3.
