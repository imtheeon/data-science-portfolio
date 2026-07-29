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
    random_baseline_precisions = []
    sample_users = test["user_id"].unique()[:50]
    for user_id in sample_users:
        if user_id not in matrix.index:
            continue
        relevant = set(test[(test["user_id"] == user_id) & (test["rating"] >= 4)]["item_id"])
        if not relevant:
            continue
        recs = recommend_for_user(matrix, similarity, user_id, n=TOP_N, k=K_NEIGHBORS)
        precisions.append(precision_at_k(recs, relevant, TOP_N))

        # Real random baseline for this same user: the expected Precision@5
        # of drawing TOP_N items uniformly at random (no replacement) from
        # the exact same candidate pool recommend_for_user draws from
        # (items in the train matrix this user hasn't rated). By linearity
        # of expectation, E[precision@k] = R / N for any k, where R is the
        # number of relevant items in the candidate pool and N is the pool
        # size -- computed analytically, not simulated, but exact.
        user_ratings = matrix.loc[user_id]
        candidate_pool = user_ratings[user_ratings == 0].index
        n_candidates = len(candidate_pool)
        if n_candidates > 0:
            n_relevant_in_pool = len(relevant & set(candidate_pool))
            random_baseline_precisions.append(n_relevant_in_pool / n_candidates)

    mean_precision = float(np.mean(precisions)) if precisions else float("nan")
    mean_random_baseline = (
        float(np.mean(random_baseline_precisions)) if random_baseline_precisions else float("nan")
    )
    print(
        f"Mean Precision@{TOP_N} over {len(precisions)} real users with "
        f"relevant held-out items: {mean_precision:.4f}"
    )
    print(
        f"Random baseline Precision@{TOP_N} over the same {len(random_baseline_precisions)} "
        f"users (expected value of {TOP_N} random picks from each user's own candidate "
        f"pool, computed analytically as R/N and averaged): {mean_random_baseline:.4f}"
    )
    if mean_random_baseline > 0:
        print(f"Model is {mean_precision / mean_random_baseline:.2f}x the random baseline.")

    print("\nExample recommendations:")
    for user_id in sample_users[:3]:
        if user_id not in matrix.index:
            continue
        recs = recommend_for_user(matrix, similarity, user_id, n=TOP_N, k=K_NEIGHBORS)
        titles = [movies.get(i, f"item {i}") for i in recs]
        print(f"  User {user_id}: {titles}")


if __name__ == "__main__":
    main()
