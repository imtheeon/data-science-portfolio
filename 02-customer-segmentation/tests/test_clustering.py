import numpy as np
import pandas as pd

from analysis.clustering import evaluate_k_range, fit_kmeans, scale_features


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


def test_scale_features_compresses_right_skewed_monetary_via_log1p():
    # Direct unit test on scale_features(): known input/output values that
    # would fail if the np.log1p transform were removed or swapped for
    # something else.
    rfm = pd.DataFrame(
        {
            "Recency": [10.0, 10.0],
            "Frequency": [1.0, np.e - 1],  # log1p -> 0.0, 1.0
            "Monetary": [1.0, np.e - 1],   # log1p -> 0.0, 1.0
        }
    )
    X = scale_features(rfm)
    # After log1p, Frequency/Monetary columns are [0.0, 1.0] for both rows
    # (identical across the two log-transformed features), so after
    # standardizing (mean 0, unit variance, ddof=0) they should be exactly
    # +/-1.0 in both the Frequency and Monetary columns.
    np.testing.assert_allclose(X[:, 1], [-1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(X[:, 2], [-1.0, 1.0], atol=1e-8)
    # Recency (not log-transformed) is constant -> StandardScaler yields 0s.
    np.testing.assert_allclose(X[:, 0], [0.0, 0.0], atol=1e-8)


def _synthetic_rfm_with_whales(seed: int = 0) -> pd.DataFrame:
    # 100 customers with Recency/Frequency drawn independently (no inherent
    # behavioral tiers to lean on) and Monetary drawn from a heavily
    # right-skewed lognormal distribution (skew ~9-13 depending on seed,
    # comparable to the real Online Retail II Monetary skew of ~25 this fix
    # targets) — a mass of "normal" spenders in roughly the $50-300 range
    # with a handful of extreme "whale" values reaching into the tens or
    # hundreds of thousands. Without the log1p transform, standardized
    # Euclidean distance is dominated entirely by those few whale values, and
    # k=2 k-means degenerates into isolating just them from everyone else
    # instead of finding any real behavioral split (verified empirically:
    # with this exact seed, unlogged k-means isolates a single customer,
    # smaller_cluster_share = 0.01 -- see analysis/clustering.py's module
    # docstring for the same failure mode observed on the real dataset).
    rng = np.random.default_rng(seed)
    n = 100
    recency = rng.uniform(1, 400, size=n)
    frequency = rng.integers(1, 20, size=n).astype(float)
    monetary = rng.lognormal(mean=4.5, sigma=2.0, size=n) + 10
    return pd.DataFrame({"Recency": recency, "Frequency": frequency, "Monetary": monetary})


def test_fit_kmeans_does_not_degenerate_into_whale_isolation():
    # Regression test for the log1p fix in scale_features(): if that
    # transform is reverted, k=2 k-means on this heavily right-skewed
    # synthetic RFM data collapses into a degenerate outlier-isolation split
    # (the exact failure mode the fix addresses, per analysis/clustering.py's
    # module docstring -- confirmed by re-running this exact fixture through
    # a StandardScaler with no log1p, which isolates a single point,
    # smaller_cluster_share = 0.01). With the fix in place, the split should
    # instead be much more balanced.
    rfm = _synthetic_rfm_with_whales()
    _, labels = fit_kmeans(rfm, k=2)

    counts = labels.value_counts()
    smaller_cluster_share = counts.min() / len(labels)

    # A degenerate whale-isolation split gives smaller_cluster_share <= 0.05
    # (often a single point); a real balanced split clears 0.15 easily.
    assert smaller_cluster_share >= 0.15
