"""Synthetic data with a known answer: prompts have a difficulty driven by their
embedding, models have fixed abilities, correctness follows the IRT formula."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from ecoroute.eval.metrics import (
    quality_report,
    route_cheapest_above,
    routing_curve,
    savings_at_quality,
)
from ecoroute.predictors import (
    CalibratedPredictor,
    IRTPredictor,
    KNNPredictor,
    MatrixFactorizationPredictor,
    ModelMeanPredictor,
    cost_matrix,
    outcome_matrix,
)

MODELS = ["tiny", "small", "medium", "large"]
ABILITY = np.array([-1.5, -0.5, 0.5, 1.5])
COST = np.array([0.001, 0.003, 0.01, 0.03])


def make_data(n, seed, dim=16, missing=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    direction = np.random.default_rng(99).normal(size=dim)
    difficulty = 3.0 * (X @ direction) / np.linalg.norm(direction) * np.sqrt(dim) / 2
    p = 1 / (1 + np.exp(-(ABILITY[None, :] - difficulty[:, None])))
    Y = (rng.random(p.shape) < p).astype(float)
    Y[rng.random(Y.shape) < missing] = np.nan
    C = np.tile(COST, (n, 1))
    return X, Y, C, difficulty


@pytest.fixture(scope="module")
def data():
    return make_data(3000, seed=0), make_data(800, seed=1, missing=0.0)


@pytest.mark.parametrize(
    "pred, min_auc",
    [
        (KNNPredictor(k=32), 0.72),
        (MatrixFactorizationPredictor(epochs=40, batch_size=512, lr=3e-3), 0.75),
        (IRTPredictor(epochs=40, batch_size=512, lr=3e-3), 0.75),
    ],
)
def test_predictors_beat_model_mean(data, pred, min_auc):
    (X_tr, Y_tr, _, _), (X_te, Y_te, _, _) = data
    base = quality_report(ModelMeanPredictor().fit(X_tr, Y_tr, MODELS).predict_proba(X_te), Y_te)
    got = quality_report(pred.fit(X_tr, Y_tr, MODELS).predict_proba(X_te), Y_te)
    assert got["auc"] > min_auc
    assert got["brier"] < base["brier"]


def test_irt_recovers_abilities_and_difficulty(data):
    (X_tr, Y_tr, _, _), (X_te, _, _, diff_te) = data
    irt = IRTPredictor(epochs=40, batch_size=512, lr=3e-3).fit(X_tr, Y_tr, MODELS)
    theta = np.array([irt.abilities()[m][0] for m in MODELS])
    # Ability scale has a free sign with dims=1; the order must match up to that sign.
    assert abs(spearmanr(theta, ABILITY).statistic) == 1.0
    rho = spearmanr(irt.difficulty(X_te), diff_te).statistic
    assert abs(rho) > 0.8


def test_route_cheapest_above():
    P = np.array([[0.9, 0.95], [0.2, 0.8], [0.1, 0.3]])
    C = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    assert route_cheapest_above(P, C, 0.7).tolist() == [0, 1, 1]  # last: none ok, most likely


def test_routing_saves_cost_at_large_model_quality(data):
    (X_tr, Y_tr, _, _), (X_te, Y_te, C_te, _) = data
    irt = IRTPredictor(epochs=40, batch_size=512, lr=3e-3).fit(X_tr, Y_tr, MODELS)
    curve = routing_curve(irt.predict_proba(X_te), Y_te, C_te, MODELS)
    oracle = curve.loc[curve.policy == "oracle"].iloc[0]
    large = curve.loc[curve.policy == "always large"].iloc[0]
    assert oracle.accuracy >= large.accuracy and oracle.cost < large.cost
    # Matching the large model's quality exactly is noisy; allow a 2-point tolerance.
    near = curve[curve.policy.str.startswith("router") & (curve.accuracy >= large.accuracy - 0.02)]
    assert (near.cost < large.cost).any()
    assert "matched" in savings_at_quality(curve, "always large")


def test_matrices_from_outcomes():
    outcomes = pd.DataFrame(
        {
            "prompt_id": ["a", "a", "b", "b", "b"],
            "model": ["x", "y", "x", "y", "y"],
            "correct": [1.0, 0.0, 0.0, 1.0, 0.0],
            "cost_usd": [0.1, 0.2, 0.1, np.nan, np.nan],
        }
    )
    Y = outcome_matrix(outcomes, ["a", "b", "c"], ["x", "y"])
    assert Y[1, 1] == 0.5  # duplicate pair averaged
    assert np.isnan(Y[2]).all()
    C = cost_matrix(outcomes, ["a", "b"], ["x", "y"])
    assert C[0].tolist() == [0.1, 0.2] and np.isnan(C[1, 1])


def test_calibration_fixes_a_miscalibrated_predictor(data):
    (X_tr, Y_tr, _, _), (X_te, Y_te, _, _) = data
    X_va, Y_va, _, _ = make_data(800, seed=2, missing=0.0)

    class Overconfident(ModelMeanPredictor):
        name = "overconfident"

        def predict_proba(self, X):
            # Push every probability toward 0 or 1 while keeping the ranking.
            P = KNNPredictor.predict_proba(self.knn, X)
            return 1 / (1 + np.exp(-8 * (P - 0.5)))

        def fit(self, X, Y, models):
            super().fit(X, Y, models)
            self.knn = KNNPredictor(k=32).fit(X, Y, models)
            return self

    raw = Overconfident().fit(X_tr, Y_tr, MODELS)
    cal = CalibratedPredictor.wrap(raw).calibrate(X_va, Y_va)
    before = quality_report(raw.predict_proba(X_te), Y_te)
    after = quality_report(cal.predict_proba(X_te), Y_te)
    assert after["ece"] < before["ece"] / 2
    assert after["brier"] < before["brier"]
