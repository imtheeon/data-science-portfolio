"""K-means clustering over standardized RFM features, with k selected by
inertia (elbow) and silhouette score rather than picked arbitrarily."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


def _scaled(rfm_df: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(rfm_df[FEATURES])


def evaluate_k_range(rfm_df: pd.DataFrame, k_range: range, random_state: int = 42) -> pd.DataFrame:
    X = _scaled(rfm_df)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(rfm_df: pd.DataFrame, k: int, random_state: int = 42) -> tuple[KMeans, pd.Series]:
    X = _scaled(rfm_df)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, pd.Series(labels, index=rfm_df.index, name="Cluster")
