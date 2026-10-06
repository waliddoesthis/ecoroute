import numpy as np
import pytest

from ecoroute.eval.metrics import quality_report
from ecoroute.graph import SkillGraph
from ecoroute.predictors import ModelMeanPredictor
from ecoroute.service import EcoRoute

MODELS = ["small", "big"]


def make(n, seed):
    """Two skills told apart by the embedding: 'math' (small model fails) and 'chat'
    (both models succeed)."""
    rng = np.random.default_rng(seed)
    skill = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    X[:, 0] += np.where(skill == 1, 3.0, -3.0)
    p = np.where(skill[:, None] == 1, [[0.2, 0.9]], [[0.9, 0.95]])
    Y = (rng.random((n, 2)) < p).astype(float)
    return X, Y, np.where(skill == 1, "math", "chat")


@pytest.fixture(scope="module")
def graph():
    X, Y, s = make(2000, 0)
    return SkillGraph(min_skill_prompts=10).fit(X, Y, MODELS, skills=s)


def test_graph_learns_skill_specific_strength(graph):
    X, Y, s = make(500, 1)
    P = graph.predict_proba(X)
    assert P[s == "math", 0].mean() < 0.4 < P[s == "chat", 0].mean()
    base = quality_report(ModelMeanPredictor().fit(X, Y, MODELS).predict_proba(X), Y)
    assert quality_report(P, Y)["brier"] < base["brier"]


def test_explanation_names_skill_and_paths(graph):
    X, _, s = make(50, 2)
    x = X[np.where(s == "math")[0][0]]
    exp = graph.explain(x)
    assert exp.skills[0][0] == "math" and exp.skills[0][1] > 0.9
    assert abs(sum(exp.levels.values()) - 1) < 1e-6
    lines = exp.lines("big")
    assert "math" in lines[0] and "big" in lines[1]
    total = sum(e[3] for e in exp.edges["big"])
    assert abs(total - graph.predict_proba(x[None])[0, 1]) < 1e-6  # paths sum to the score


def test_small_cells_shrink_toward_model_mean():
    X, Y, s = make(400, 3)
    g = SkillGraph(shrink=1e6, min_skill_prompts=10).fit(X, Y, MODELS, skills=s)
    assert np.allclose(g.acc_, np.nanmean(Y, axis=0), atol=1e-3)


def test_needs_skill_labels():
    with pytest.raises(ValueError):
        SkillGraph().fit(np.zeros((4, 2)), np.zeros((4, 2)), MODELS)


def test_service_puts_graph_paths_in_the_explanation(graph):
    catalog = [
        {"name": "cheap", "tier": 1, "clearance": "internal", "price_out_per_mtok": 1.0,
         "anchor": {"model": "small"}},
        {"name": "premium", "tier": 3, "clearance": "internal", "price_out_per_mtok": 20.0,
         "anchor": {"model": "big"}},
    ]  # fmt: skip
    eco = EcoRoute(graph, "fake", catalog)
    X, _, s = make(50, 4)
    math_x, chat_x = X[s == "math"][0], X[s == "chat"][0]
    eco.embed = lambda texts: (math_x if texts[0] == "math q" else chat_x)[None]
    d = eco.route("math q")
    assert d.model == "premium"
    assert any("premium (via big)" in step for step in d.steps)
    assert eco.route("hello").model == "cheap"


def test_labelled_prompts_feed_only_the_difficulty_edge():
    X, Y, s = make(600, 5)
    rng = np.random.default_rng(6)
    X2 = rng.normal(size=(300, 8)).astype(np.float32)
    X2[:, 1] += 4.0  # a direction the outcome data never shows as hard
    g = SkillGraph(min_skill_prompts=10).fit(
        X, Y, MODELS, skills=s, level_extra=(X2, np.full(300, 2))
    )
    probe = np.zeros((1, 8), np.float32)
    probe[0, 1] = 4.0
    assert g.explain(probe[0]).levels["hard"] > 0.5
    plain = SkillGraph(min_skill_prompts=10).fit(X, Y, MODELS, skills=s)
    assert np.allclose(g.acc_, plain.acc_)  # model edges come from outcomes only


def test_similar_prompt_edges_and_beta():
    X, Y, s = make(1500, 7)
    g = SkillGraph(min_skill_prompts=10, k=16).fit(X, Y, MODELS, skills=s)
    Xv, Yv, _ = make(400, 8)
    beta = g.tune_beta(Xv, Yv)
    assert 0.0 <= beta <= 1.0
    x = Xv[0]
    exp = g.explain(x)
    paths = sum(e[3] for e in exp.edges["big"])
    near = exp.neighbours.get("big", (0, 0, 0.0))[2]
    assert abs(paths + near - g.predict_proba(x[None])[0, 1]) < 1e-6  # all edges sum to the score
    if beta > 0:
        solved, answered, _ = exp.neighbours["big"]
        assert 0 <= solved <= answered <= 16
        assert "most similar past prompts" in exp.lines("big")[1]
