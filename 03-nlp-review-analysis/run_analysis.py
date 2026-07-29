# run_analysis.py
"""End-to-end run: load a real Amazon review sample -> VADER sentiment ->
compare against real star ratings -> NMF topic model -> save charts and
model artifacts for the Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from analysis.plots import plot_sentiment_crosstab, plot_topic_volume
from analysis.sentiment import evaluate_agreement, rating_to_label, score_sentiment
from analysis.text_cleaning import strip_html
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

    # Real review text contains literal HTML (mostly stray `<br />` tags).
    # Clean it once here, before it's used for both sentiment scoring and
    # topic modeling, so the two stay consistent and HTML markup doesn't
    # get vectorized as if it were content (see analysis/text_cleaning.py).
    df["Text"] = df["Text"].apply(strip_html)

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

    fig = plot_sentiment_crosstab(crosstab, agreement["accuracy"])
    fig.savefig(CHARTS_DIR / "sentiment_vs_rating.png", dpi=150)
    plt.close(fig)

    model, vectorizer, topic_words, W = fit_topic_model(df["Text"].tolist(), n_topics=N_TOPICS)
    df["topic"] = dominant_topics(W)

    for i, words in enumerate(topic_words):
        print(f"Topic {i}: {', '.join(words)}")

    topic_sizes = df["topic"].value_counts().sort_index()
    print("Reviews per topic:")
    print(topic_sizes)
    fig2 = plot_topic_volume(topic_sizes, topic_words)
    fig2.savefig(CHARTS_DIR / "topics.png", dpi=150)
    plt.close(fig2)

    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "nmf_model.joblib")
    joblib.dump(topic_words, MODEL_DIR / "topic_words.joblib")

    df.to_csv(CHARTS_DIR.parent / "analyzed_sample.csv", index=False)
    print("Charts saved to charts/, model artifacts saved to model/, full sample saved to analyzed_sample.csv")


if __name__ == "__main__":
    main()
