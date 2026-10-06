"""Shared interface: given prompt embeddings, predict P(correct) for every model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def outcome_matrix(outcomes: pd.DataFrame, prompt_ids: list[str], models: list[str]) -> np.ndarray:
    """(n_prompts, n_models) correctness, NaN where a model never answered that prompt.

    When a prompt/model pair appears in more than one source, the scores are averaged.
    """
    table = outcomes.pivot_table(
        index="prompt_id", columns="model", values="correct", aggfunc="mean"
    )
    return table.reindex(index=prompt_ids, columns=models).to_numpy(dtype=np.float64)


def cost_matrix(outcomes: pd.DataFrame, prompt_ids: list[str], models: list[str]) -> np.ndarray:
    """(n_prompts, n_models) observed USD cost per answer, NaN where unknown."""
    table = outcomes.pivot_table(
        index="prompt_id", columns="model", values="cost_usd", aggfunc="mean"
    )
    return table.reindex(index=prompt_ids, columns=models).to_numpy(dtype=np.float64)


class Predictor(ABC):
    name: str = "predictor"

    def __init__(self) -> None:
        self.models: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, models: list[str]) -> "Predictor":
        """X: (n, d) embeddings. Y: (n, m) correctness in [0, 1], NaN when unobserved."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, m) predicted probability that each model answers correctly."""


class ModelMeanPredictor(Predictor):
    """Reference baseline: every prompt gets each model's average accuracy."""

    name = "model_mean"

    def fit(self, X, Y, models):
        self.models = list(models)
        self.mean_ = np.nanmean(Y, axis=0)
        return self

    def predict_proba(self, X):
        return np.tile(self.mean_, (len(X), 1))
