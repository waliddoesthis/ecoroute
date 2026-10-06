"""Predict for the deployment catalog using a predictor trained on public outcome data.

The public datasets grade older models (gpt-4o, llama-3.1, ...), not the ones we deploy.
Each catalog model is anchored to a training model plus a logit shift:

    logit P_catalog(q) = logit P_anchor(q) + shift

The anchor carries the per-prompt pattern (which prompts are hard, which are code, ...);
the shift says how much stronger or weaker the deployed model is across the board. A shift
of 0 assumes the deployed model is exactly as good as its anchor, which for newer models
underestimates them, so the router errs toward premium models (safe for quality, costly).
fit_shift() replaces that prior with a value fitted on a few hundred graded answers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecoroute.predictors.base import Predictor

_EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass
class Anchor:
    model: str
    shift: float = 0.0
    source: str = "prior"  # "prior" until fit_shift() sets it to "fitted"


class CatalogPredictor:
    def __init__(self, base: Predictor, anchors: Mapping[str, Anchor]) -> None:
        missing = {a.model for a in anchors.values()} - set(base.models)
        if missing:
            raise ValueError(f"anchors not in the trained predictor: {sorted(missing)}")
        self.base = base
        self.anchors = dict(anchors)
        self.names = list(self.anchors)

    @classmethod
    def from_catalog(cls, base: Predictor, catalog: list[Mapping[str, Any]]) -> CatalogPredictor:
        """Read each model's `anchor: {model, shift}` from configs/models.yaml entries."""
        anchors = {}
        for m in catalog:
            a = m.get("anchor")
            if a:
                anchors[m["name"]] = Anchor(
                    a["model"], float(a.get("shift", 0.0)), a.get("source", "prior")
                )
        return cls(base, anchors)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, len(names)) P(correct) for the catalog models."""
        P = self.base.predict_proba(X)
        idx = {m: j for j, m in enumerate(self.base.models)}
        cols = [
            _sigmoid(_logit(P[:, idx[a.model]]) + a.shift)
            for a in (self.anchors[n] for n in self.names)
        ]
        return np.stack(cols, axis=1)

    def predict_one(self, x: np.ndarray) -> dict[str, float]:
        p = self.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
        return {n: float(v) for n, v in zip(self.names, p)}

    def fit_shift(self, name: str, X: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
        """Fit the shift for `name` by maximum likelihood on graded answers y in [0, 1].

        With the anchor's logits fixed, this is a one-parameter logistic regression
        (an offset), solved with Newton's method.
        """
        a = self.anchors[name]
        j = self.base.models.index(a.model)
        z = _logit(self.base.predict_proba(X)[:, j])
        y = np.asarray(y, dtype=np.float64)
        keep = ~np.isnan(y)
        z, y = z[keep], y[keep]
        shift = 0.0
        for _ in range(iters):
            p = _sigmoid(z + shift)
            grad = np.sum(y - p)
            hess = np.sum(p * (1 - p)) + 1e-9
            step = grad / hess
            shift += step
            if abs(step) < 1e-8:
                break
        a.shift, a.source = float(shift), "fitted"
        return a.shift
