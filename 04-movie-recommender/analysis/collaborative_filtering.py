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
