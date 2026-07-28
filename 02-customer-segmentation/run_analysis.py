# run_analysis.py
"""End-to-end run: load real data -> RFM -> pick k -> cluster -> profile ->
label -> save charts. Prints the real numbers that go into the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from analysis.clustering import evaluate_k_range, fit_kmeans, scale_features
from analysis.rfm import compute_rfm
from analysis.segment_actions import label_all_clusters, profile_clusters
from data.load_online_retail import load_raw

CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw transaction rows.")

    rfm = compute_rfm(raw)
    print(f"Computed RFM for {len(rfm):,} real customers.")

    k_scores = evaluate_k_range(rfm, range(2, 9))
    print(k_scores.to_string(index=False))
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
    print(f"Selected k={best_k} (highest silhouette score).")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(k_scores["k"], k_scores["inertia"], marker="o")
    ax[0].set_title("Elbow: Inertia vs. k")
    ax[0].set_xlabel("k")
    ax[1].plot(k_scores["k"], k_scores["silhouette"], marker="o", color="darkorange")
    ax[1].set_title("Silhouette Score vs. k")
    ax[1].set_xlabel("k")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "k_selection.png", dpi=150)
    plt.close(fig)

    _, labels = fit_kmeans(rfm, k=best_k)
    rfm_clustered = rfm.copy()
    rfm_clustered["Cluster"] = labels

    # Real median Recency/Frequency/Monetary across all individual customers
    # (before/regardless of clustering) — the baseline label_all_clusters
    # benchmarks each cluster against, per the README's documented intent.
    overall_medians = rfm_clustered[["Recency", "Frequency", "Monetary"]].median()

    profile = profile_clusters(rfm_clustered)
    profile = label_all_clusters(profile, overall_medians)
    print(profile.to_string(index=False))
    profile.to_csv(CHARTS_DIR.parent / "cluster_profile.csv", index=False)

    X_scaled = scale_features(rfm_clustered)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=rfm_clustered["Cluster"], cmap="tab10", alpha=0.6, s=15)
    ax2.set_title(f"Customer Segments (PCA projection, k={best_k})")
    legend = ax2.legend(*scatter.legend_elements(), title="Cluster")
    ax2.add_artist(legend)
    fig2.tight_layout()
    fig2.savefig(CHARTS_DIR / "pca_clusters.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.barh(profile["Segment"], profile["PctRevenue"])
    ax3.set_xlabel("% of total revenue")
    ax3.set_title("Revenue Share by Segment")
    fig3.tight_layout()
    fig3.savefig(CHARTS_DIR / "revenue_by_segment.png", dpi=150)
    plt.close(fig3)

    print("Charts saved to charts/. Cluster profile saved to cluster_profile.csv.")


if __name__ == "__main__":
    main()
