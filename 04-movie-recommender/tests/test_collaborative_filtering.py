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
