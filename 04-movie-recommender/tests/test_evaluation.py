import pytest

from analysis.evaluation import precision_at_k, rmse


def test_rmse_perfect_predictions():
    assert rmse([3, 4, 5], [3, 4, 5]) == 0.0


def test_rmse_known_value():
    assert rmse([4, 4], [3, 5]) == pytest.approx(1.0)


def test_precision_at_k_partial_hits():
    assert precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5) == pytest.approx(3 / 5)


def test_precision_at_k_no_hits():
    assert precision_at_k([10, 20], {1, 2, 3}, k=2) == 0.0


def test_precision_at_k_empty_recommendations():
    assert precision_at_k([], {1, 2}, k=5) == 0.0
