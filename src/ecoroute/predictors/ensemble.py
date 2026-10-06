"""Average of several fitted predictors.

kNN, matrix factorization and IRT make different mistakes (local neighbours vs. global
model profiles vs. a difficulty scale), so their average is usually better than each
one, especially on hard prompts where single predictors disagree most.
"""

from __future__ import annotations

import numpy as np

from ecoroute.predictors.base import Predictor


class EnsemblePredictor(Predictor):
    def __init__(self, members: list[Predictor], weights: list[float] | None = None) -> None:
        super().__init__()
        self.members = members
        self.weights = np.asarray(weights if weights else [1.0] * len(members), dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.name = "ensemble(" + "+".join(m.name for m in members) + ")"

    def fit(self, X, Y, models):
        for m in self.members:
            m.fit(X, Y, models)
        self.models = list(models)
        return self

    @classmethod
    def of_fitted(cls, members: list[Predictor], **kw) -> "EnsemblePredictor":
        ens = cls(members, **kw)
        ens.models = members[0].models
        if any(m.models != ens.models for m in members):
            raise ValueError("ensemble members must share the same model order")
        return ens

    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.members))
