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
    return pd.DataFrame(rows, columns=["compound", "predicted_label"])


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
