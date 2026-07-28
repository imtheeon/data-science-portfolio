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
