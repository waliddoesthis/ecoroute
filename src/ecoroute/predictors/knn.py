"""kNN over prompt embeddings: "how did each model do on the most similar past prompts?"

The 2026 Routing Plateau study found this simple method competitive with most learned
routers, so every other predictor has to beat it. It also gives example-based
explanations through neighbours().
"""

from __future__ import annotations

import numpy as np

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
        self.unit_ = _unit(X)
        self._last = None
        return self

    def neighbours(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(similarities, row indices into the training set), each (n, k), most similar first.

        Exact cosine search as one matrix product over unit vectors kept from fit. The
        router asks several times about the same prompt (prediction, edge weight,
        explanation), so the last answer is reused.
        """
        if getattr(self, "unit_", None) is None:  # routers saved before unit_ existed
            self.unit_ = _unit(self.index_._fit_X)
        X = np.asarray(X, dtype=np.float32)
        key = X.tobytes()
        last = getattr(self, "_last", None)
        if last is not None and last[0] == key:
            return last[1]
        k = min(self.k, len(self.unit_))
        sims, idxs = [], []
        for start in range(0, len(X), 1024):  # bounds memory for large batches
            sim = _unit(X[start : start + 1024]) @ self.unit_.T
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            top = np.take_along_axis(sim, idx, axis=1)
            order = np.argsort(-top, axis=1)
            idxs.append(np.take_along_axis(idx, order, axis=1))
            sims.append(np.take_along_axis(top, order, axis=1))
        out = (np.concatenate(sims).astype(np.float64), np.concatenate(idxs))
        self._last = (key, out)
        return out

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_last", None)
        state.pop("index_", None)  # unit_ replaces the sklearn index
        return state

    def predict_proba(self, X):
        sim, idx = self.neighbours(X)
        w = np.clip(sim, 0.0, None)[:, :, None]  # (n, k, 1)
        y = self.Y_[idx]  # (n, k, m)
        seen = ~np.isnan(y)
        num = np.where(seen, y * w, 0.0).sum(axis=1)
        den = np.where(seen, w, 0.0).sum(axis=1)
        return (num + self.prior_weight * self.prior_) / (den + self.prior_weight)


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
