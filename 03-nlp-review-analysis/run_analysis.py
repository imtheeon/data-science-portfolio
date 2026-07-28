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
