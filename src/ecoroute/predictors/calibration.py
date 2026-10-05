"""Per-model calibration on held-out data.

The router compares P(correct) against a threshold, so a model whose raw scores run
high or low shifts every routing decision. Isotonic regression per model, fitted on the
validation split, maps raw scores to observed accuracy without assuming a shape.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ecoroute.predictors.base import Predictor


class CalibratedPredictor(Predictor):
    def __init__(self, base: Predictor, min_samples: int = 50) -> None:
        super().__init__()
        self.base = base
        # Models with fewer validation answers than this keep their raw scores.
        self.min_samples = min_samples
        self.name = f"{base.name}+calibrated"

    def fit(self, X, Y, models):
        self.base.fit(X, Y, models)
        self.models = self.base.models
        self.calibrators_ = [None] * len(self.models)
        return self

    @classmethod
    def wrap(cls, fitted: Predictor, **kw) -> "CalibratedPredictor":
        """Wrap an already-fitted predictor without training it again."""
        cal = cls(fitted, **kw)
        cal.models = fitted.models
        cal.calibrators_ = [None] * len(cal.models)
        return cal

    def calibrate(self, X_val: np.ndarray, Y_val: np.ndarray) -> "CalibratedPredictor":
        """Fit one isotonic map per model on validation predictions vs. observed scores."""
        raw = self.base.predict_proba(X_val)
        for j in range(len(self.models)):
            seen = ~np.isnan(Y_val[:, j])
            if seen.sum() < self.min_samples:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrators_[j] = iso.fit(raw[seen, j], Y_val[seen, j])
        return self

    def predict_proba(self, X):
        P = self.base.predict_proba(X).copy()
        for j, iso in enumerate(self.calibrators_):
            if iso is not None:
                P[:, j] = iso.predict(P[:, j])
        return P
