"""Adapt the router to your own work with outcomes graded on your own prompts.

The trained graph knows public benchmarks, and each catalog model is placed through an
older benchmark model plus a prior shift (catalog.py). Your own graded prompts ("on this
prompt, claude-haiku-4-5 was good enough: yes/no") correct both:

1. Shift: for every model with enough graded answers, the logit shift is refit by maximum
   likelihood, so the model's overall strength on your work is measured, not assumed.
2. Similar prompts: for a new prompt, the graded prompts most like it pull the prediction
   toward what actually happened on them:

       P = (sum_i w_i * y_i + strength * P_prior) / (sum_i w_i + strength)

   with w_i the cosine similarity of the i-th nearest graded prompt for that model (only
   neighbours above min_similarity count). With no close neighbours, P stays the prior.

Both are fitted on a training part of your data and checked on the rest (see
scripts/retrain.py), so you can see whether they help before deploying them.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class LocalOutcomes:
    def __init__(
        self,
        X: np.ndarray,
        models: list[str],
        y: np.ndarray,
        k: int = 16,
        strength: float = 4.0,
        min_similarity: float = 0.5,
    ) -> None:
        self.unit_ = _unit(X)
        self.models_ = np.asarray(models)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.k, self.strength, self.min_similarity = k, strength, min_similarity
        self.counts = {m: int((self.models_ == m).sum()) for m in sorted(set(models))}
        self._last: tuple[bytes, dict] | None = None

    def evidence(self, x: np.ndarray) -> dict[str, tuple[float, float, int]]:
        """Per model: (weight sum, weighted success, neighbours used) for prompt x."""
        x = _unit(np.asarray(x).reshape(1, -1))[0]
        key = x.tobytes()
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        sim = self.unit_ @ x
        out = {}
        for m in self.counts:
            idx = np.flatnonzero(self.models_ == m)
            s = sim[idx]
            top = np.argsort(-s)[: self.k]
            keep = top[s[top] >= self.min_similarity]
            if len(keep):
                w = s[keep]
                out[m] = (float(w.sum()), float(w @ self.y_[idx[keep]]), len(keep))
        self._last = (key, out)
        return out

    def adjust(self, x: np.ndarray, p_prior: Mapping[str, float | None]) -> dict:
        out = dict(p_prior)
        for m, (wsum, wy, _) in self.evidence(x).items():
            prior = p_prior.get(m)
            if prior is None:
                continue
            out[m] = (wy + self.strength * prior) / (wsum + self.strength)
        return out

    def explain(self, x: np.ndarray, model: str) -> str | None:
        ev = self.evidence(x).get(model)
        if ev is None:
            return None
        wsum, wy, n = ev
        return (
            f"Your data: {model} handled {wy / wsum:.0%} of the {n} most similar prompts "
            f"you graded (out of {self.counts[model]} for it)."
        )


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
