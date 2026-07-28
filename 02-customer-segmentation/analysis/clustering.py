"""K-means clustering over standardized RFM features, with k selected by
inertia (elbow) and silhouette score rather than picked arbitrarily.

Frequency and Monetary are heavily right-skewed in real retail transaction
data (a handful of "whale" customers can be 1-2 orders of magnitude above
the median). Left unlogged, Euclidean distance on standardized-but-skewed
features is dominated by those outliers, so k-means just isolates the whales
from everyone else instead of surfacing differentiated behavioral segments.
log1p-transforming Frequency and Monetary before scaling is standard
practice for RFM clustering and fixes this."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]
LOG_FEATURES = ["Frequency", "Monetary"]


def scale_features(rfm_df: pd.DataFrame):
    transformed = rfm_df[FEATURES].copy()
    for col in LOG_FEATURES:
        transformed[col] = np.log1p(transformed[col])
    scaler = StandardScaler()
    return scaler.fit_transform(transformed)


def evaluate_k_range(rfm_df: pd.DataFrame, k_range: range, random_state: int = 42) -> pd.DataFrame:
    X = scale_features(rfm_df)
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
    X = scale_features(rfm_df)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, pd.Series(labels, index=rfm_df.index, name="Cluster")
