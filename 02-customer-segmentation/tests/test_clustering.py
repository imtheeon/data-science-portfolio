import numpy as np
import pandas as pd

from analysis.clustering import evaluate_k_range, fit_kmeans


def _synthetic_rfm(seed: int = 0) -> pd.DataFrame:
    # Three well-separated blobs so clustering behavior is unambiguous to test.
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[5, 5, 5], scale=0.5, size=(30, 3))
    blob_b = rng.normal(loc=[50, 50, 50], scale=0.5, size=(30, 3))
    blob_c = rng.normal(loc=[100, 1, 100], scale=0.5, size=(30, 3))
    data = np.vstack([blob_a, blob_b, blob_c])
    return pd.DataFrame(data, columns=["Recency", "Frequency", "Monetary"])


def test_evaluate_k_range_returns_expected_shape():
    rfm = _synthetic_rfm()
    result = evaluate_k_range(rfm, range(2, 6))
    assert list(result["k"]) == [2, 3, 4, 5]
    assert (result["inertia"] > 0).all()
    assert result["silhouette"].between(-1, 1).all()


def test_fit_kmeans_recovers_three_blobs():
    rfm = _synthetic_rfm()
    model, labels = fit_kmeans(rfm, k=3)
    assert labels.nunique() == 3
    assert len(labels) == len(rfm)
    # Each of the three well-separated blobs should end up as its own cluster.
    counts = labels.value_counts()
    assert (counts == 30).all()
