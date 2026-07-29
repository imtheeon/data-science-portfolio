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
