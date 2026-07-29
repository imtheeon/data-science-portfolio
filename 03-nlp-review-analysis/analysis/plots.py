"""Shared matplotlib chart-building functions, used by both
run_analysis.py and notebooks/review_analysis.ipynb so the two don't
duplicate plotting code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_sentiment_crosstab(crosstab: pd.DataFrame, accuracy: float) -> plt.Figure:
    """Confusion-matrix-style heatmap of star-rating-derived label vs.
    VADER-predicted label."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(crosstab.values, cmap="Blues")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index)
    ax.set_xlabel("VADER predicted label")
    ax.set_ylabel("Star-rating-derived label")
    ax.set_title(f"Sentiment vs. Rating Agreement (accuracy={accuracy:.2%})", fontsize=11)
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax.text(j, i, crosstab.values[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    return fig


def plot_topic_volume(topic_sizes: pd.Series, topic_words: list[list[str]]) -> plt.Figure:
    """Horizontal bar chart of review count per topic, labeled with each
    topic's top-3 words."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"Topic {i}: {', '.join(topic_words[i][:3])}" for i in topic_sizes.index]
    ax.barh(labels, topic_sizes.values)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Review Volume by Topic")
    fig.tight_layout()
    return fig
