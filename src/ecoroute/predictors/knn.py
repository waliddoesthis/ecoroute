"""kNN over prompt embeddings: "how did each model do on the most similar past prompts?"

The 2026 Routing Plateau study found this simple method competitive with most learned
routers, so every other predictor has to beat it. It also gives example-based
explanations through neighbours().
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ecoroute.predictors.base import Predictor


class KNNPredictor(Predictor):
    name = "knn"

    def __init__(self, k: int = 32, prior_weight: float = 1.0) -> None:
        super().__init__()
        self.k = k
        # Shrinks toward the model's global accuracy when few neighbours have an answer.
        self.prior_weight = prior_weight

    def fit(self, X, Y, models):
        self.models = list(models)
        self.Y_ = Y
        self.prior_ = np.nanmean(Y, axis=0)
        self.index_ = NearestNeighbors(n_neighbors=min(self.k, len(X)), metric="cosine").fit(X)
        return self

    def neighbours(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(similarities, row indices into the training set), each (n, k)."""
        dist, idx = self.index_.kneighbors(X)
        return 1.0 - dist, idx

    def predict_proba(self, X):
        sim, idx = self.neighbours(X)
        w = np.clip(sim, 0.0, None)[:, :, None]  # (n, k, 1)
        y = self.Y_[idx]  # (n, k, m)
        seen = ~np.isnan(y)
        num = np.where(seen, y * w, 0.0).sum(axis=1)
        den = np.where(seen, w, 0.0).sum(axis=1)
        return (num + self.prior_weight * self.prior_) / (den + self.prior_weight)
