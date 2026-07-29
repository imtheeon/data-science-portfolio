"""Standard recommender-system evaluation metrics: RMSE for rating
prediction accuracy, Precision@K for top-N recommendation quality."""

from __future__ import annotations

import numpy as np


def rmse(predictions: list[float], actuals: list[float]) -> float:
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)
