import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

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


def test_shrinkage_penalizes_low_support_similarity():
    # Regression test for the shrinkage-regularization fix: item pair (1, 2)
    # is co-rated by exactly ONE user, item pair (3, 4) is co-rated by FIVE
    # users -- but both pairs have identical, "perfect" raw cosine
    # similarity of 1.0 (each pair's ratings are proportional across every
    # user who rated them). Without shrinkage these would be
    # indistinguishable; that's exactly the bug shrinkage fixes.
    data = [
        (1, 1, 4), (1, 2, 4),          # only shared rater for items 1 & 2
        (2, 3, 3), (2, 4, 3),
        (3, 3, 4), (3, 4, 4),
        (4, 3, 2), (4, 4, 2),
        (5, 3, 5), (5, 4, 5),
        (6, 3, 1), (6, 4, 1),          # 5 shared raters for items 3 & 4
    ]
    ratings = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
    matrix = build_user_item_matrix(ratings)

    # Sanity check the fixture: raw (unshrunk) cosine similarity really is
    # a "perfect" 1.0 for both pairs despite wildly different support.
    raw = pd.DataFrame(
        cosine_similarity(matrix.T.values), index=matrix.columns, columns=matrix.columns
    )
    assert raw.loc[1, 2] == pytest.approx(1.0)
    assert raw.loc[3, 4] == pytest.approx(1.0)

    shrunk = compute_item_similarity(matrix)

    # The 1-shared-rater pair must be shrunk well below its "perfect" raw
    # cosine angle -- a single co-rater is not real evidence of similarity.
    assert shrunk.loc[1, 2] < 0.1

    # The better-supported pair (5 shared raters) should be shrunk less and
    # end up meaningfully higher than the 1-rater pair, even though both
    # have identical raw cosine similarity. If shrinkage were reverted
    # (e.g. beta=0, no shrinkage), both pairs would collapse to the same
    # raw 1.0 similarity and this assertion would fail.
    assert shrunk.loc[3, 4] > shrunk.loc[1, 2]


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
