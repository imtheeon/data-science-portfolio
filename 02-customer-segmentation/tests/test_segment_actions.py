import pytest
import pandas as pd

from analysis.segment_actions import label_all_clusters, label_segment, profile_clusters


def test_profile_clusters_computes_real_shares():
    rfm = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Recency": [5, 5, 100, 100],
            "Frequency": [10, 10, 1, 1],
            "Monetary": [1000.0, 1000.0, 10.0, 10.0],
            "Cluster": [0, 0, 1, 1],
        }
    )
    profile = profile_clusters(rfm)
    cluster0 = profile[profile["Cluster"] == 0].iloc[0]
    assert cluster0["CustomerCount"] == 2
    assert cluster0["PctCustomers"] == 50.0
    assert cluster0["PctRevenue"] == pytest.approx(2000.0 / 2020.0 * 100, rel=1e-6)


def test_label_segment_champions_vs_at_risk():
    overall = pd.Series({"Recency": 50, "Frequency": 3, "Monetary": 200.0})

    champion_row = pd.Series({"Recency": 5, "Frequency": 12, "Monetary": 1500.0})
    name, action = label_segment(champion_row, overall)
    assert name == "Champions"
    assert "loyalty" in action.lower() or "reward" in action.lower()

    at_risk_row = pd.Series({"Recency": 200, "Frequency": 1, "Monetary": 20.0})
    name2, action2 = label_segment(at_risk_row, overall)
    assert name2 in {"At Risk", "Hibernating"}
    assert "win-back" in action2.lower() or "re-engage" in action2.lower()


def test_label_all_clusters_uses_the_provided_overall_medians():
    # profile holds per-cluster means; overall_medians here is deliberately
    # NOT the median of these two rows (that would be the tautological
    # cluster-mean-of-means baseline the fix replaced) — it represents the
    # real customer-base median, which the caller computes separately and
    # passes in. label_all_clusters must use exactly the medians it's given.
    profile = pd.DataFrame(
        {
            "Cluster": [0, 1],
            "CustomerCount": [2, 2],
            "Recency": [10.0, 300.0],
            "Frequency": [10.0, 1.0],
            "Monetary": [2000.0, 50.0],
        }
    )
    overall_medians = pd.Series({"Recency": 50.0, "Frequency": 2.0, "Monetary": 300.0})

    labeled = label_all_clusters(profile, overall_medians)

    assert list(labeled["Segment"]) == ["Champions", "Hibernating"]
    assert "MarketingAction" in labeled.columns
    assert len(labeled) == 2
