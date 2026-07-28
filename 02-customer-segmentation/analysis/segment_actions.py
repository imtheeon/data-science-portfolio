"""Turns real cluster statistics into named segments with a concrete
marketing action — the mapping is a heuristic over each cluster's actual
computed Recency/Frequency/Monetary relative to the overall customer base,
not an arbitrary label."""

from __future__ import annotations

import pandas as pd


def profile_clusters(rfm_with_clusters: pd.DataFrame) -> pd.DataFrame:
    total_customers = len(rfm_with_clusters)
    total_revenue = rfm_with_clusters["Monetary"].sum()

    rows = []
    for cluster_id, group in rfm_with_clusters.groupby("Cluster"):
        rows.append(
            {
                "Cluster": cluster_id,
                "CustomerCount": len(group),
                "PctCustomers": len(group) / total_customers * 100,
                "PctRevenue": group["Monetary"].sum() / total_revenue * 100,
                "Recency": group["Recency"].mean(),
                "Frequency": group["Frequency"].mean(),
                "Monetary": group["Monetary"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("PctRevenue", ascending=False).reset_index(drop=True)


def label_segment(cluster_row: pd.Series, overall_medians: pd.Series) -> tuple[str, str]:
    recent = cluster_row["Recency"] <= overall_medians["Recency"]
    frequent = cluster_row["Frequency"] >= overall_medians["Frequency"]
    high_value = cluster_row["Monetary"] >= overall_medians["Monetary"]

    if recent and frequent and high_value:
        return "Champions", "Enroll in a loyalty/VIP rewards program to protect this high-value relationship."
    if recent and frequent:
        return "Loyal Customers", "Upsell/cross-sell campaigns — they buy often, grow their basket size."
    if recent and not frequent:
        return "New / Promising", "Onboarding nurture campaign to convert first purchase into a habit."
    if not recent and (frequent or high_value):
        return "At Risk", "Targeted win-back offer before they churn — they used to be valuable."
    return "Hibernating", "Low-cost re-engage email; deprioritize spend versus other segments."


def label_all_clusters(profile: pd.DataFrame, overall_medians: pd.Series) -> pd.DataFrame:
    """Label every cluster in `profile` against `overall_medians` — the real
    median Recency/Frequency/Monetary computed across all individual
    customers in the RFM DataFrame (before/regardless of clustering), not a
    median of the per-cluster means. Callers (see run_analysis.py) must
    compute that baseline from the ungrouped customer-level data and pass it
    in here."""
    labels = profile.apply(lambda row: label_segment(row, overall_medians), axis=1)
    profile = profile.copy()
    profile["Segment"] = [l[0] for l in labels]
    profile["MarketingAction"] = [l[1] for l in labels]
    return profile
