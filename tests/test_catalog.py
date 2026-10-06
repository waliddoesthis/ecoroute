import numpy as np
import pytest
import yaml

from ecoroute.predictors import Anchor, CatalogPredictor, KNNPredictor, ModelMeanPredictor
from ecoroute.service import EcoRoute
from tests.test_predictors import MODELS, make_data


@pytest.fixture(scope="module")
def trained():
    X, Y, _, _ = make_data(1500, seed=0)
    return KNNPredictor(k=32).fit(X, Y, MODELS)


def test_shift_zero_copies_the_anchor(trained):
    cat = CatalogPredictor(trained, {"new": Anchor("medium")})
    X, *_ = make_data(50, seed=3)
    assert np.allclose(
        cat.predict_proba(X)[:, 0], np.clip(trained.predict_proba(X)[:, 2], 1e-4, 1 - 1e-4)
    )


def test_positive_shift_raises_every_prompt(trained):
    X, *_ = make_data(50, seed=3)
    cat = CatalogPredictor(trained, {"same": Anchor("medium"), "better": Anchor("medium", 1.0)})
    P = cat.predict_proba(X)
    assert (P[:, 1] > P[:, 0]).all()


def test_fit_shift_recovers_a_known_offset(trained):
    X, *_ = make_data(4000, seed=4, missing=0.0)
    cat = CatalogPredictor(trained, {"new": Anchor("small")})
    z = np.log(np.clip(trained.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4))
    z -= np.log(1 - np.clip(trained.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4))
    true_p = 1 / (1 + np.exp(-(z + 0.8)))
    y = (np.random.default_rng(0).random(len(X)) < true_p).astype(float)
    assert abs(cat.fit_shift("new", X, y) - 0.8) < 0.15
    assert cat.anchors["new"].source == "fitted"


def test_unknown_anchor_is_rejected(trained):
    with pytest.raises(ValueError, match="not in the trained predictor"):
        CatalogPredictor(trained, {"x": Anchor("gpt-99")})


def test_catalog_anchors_point_at_sprout_models():
    sprout = {
        "claude-3.5-sonnet", "gpt-4o", "gpt-4o-mini", "granite-3-2b-instruct",
        "granite-3-8b-instruct", "llama-3.1-405b-instruct", "llama-3.1-70b-instruct",
        "llama-3.1-8b-instruct", "llama-3.2-1b-instruct", "llama-3.2-3b-instruct",
        "llama-3.3-70b-instruct", "mixtral-8x7b-instruct", "titan-text-premier",
    }  # fmt: skip
    catalog = yaml.safe_load(open("configs/models.yaml"))["models"]
    assert all(m["anchor"]["model"] in sprout for m in catalog)


def test_service_round_trip(tmp_path, monkeypatch):
    catalog = [
        {"name": "local", "tier": 0, "clearance": "restricted", "anchor": {"model": "a"}},
        {"name": "cloud", "tier": 2, "clearance": "internal", "price_out_per_mtok": 5.0,
         "anchor": {"model": "b"}},
    ]  # fmt: skip
    base = ModelMeanPredictor().fit(np.zeros((4, 3)), np.array([[0, 1]] * 4, float), ["a", "b"])
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": catalog}))
    EcoRoute(base, "fake-encoder", catalog).save(tmp_path / "router.pt")
    eco = EcoRoute.load(tmp_path / "router.pt", catalog=tmp_path / "models.yaml")
    monkeypatch.setattr(eco, "embed", lambda texts: np.zeros((len(texts), 3), np.float32))
    assert eco.route("what is 2+2?").model == "cloud"
    assert eco.route("hi", context={"min_level": "restricted"}).model == "local"
    # The email is found by the privacy check that runs beside the embedding.
    assert eco.route("mail jane.doe@example.com the notes").model == "local"


def test_service_saves_tuned_taus(tmp_path):
    catalog = [{"name": "local", "tier": 0, "clearance": "restricted", "anchor": {"model": "a"}}]
    base = ModelMeanPredictor().fit(np.zeros((4, 3)), np.array([[1.0]] * 4), ["a"])
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": catalog}))
    EcoRoute(base, "fake", catalog, taus={"balanced": 0.93}).save(tmp_path / "r.pt")
    eco = EcoRoute.load(tmp_path / "r.pt", catalog=tmp_path / "models.yaml")
    assert eco.router.policy.tau == 0.93
    assert eco.router.profiles["quality"].tau == 0.93


def test_route_reports_stage_timings(tmp_path, monkeypatch):
    catalog = [{"name": "local", "tier": 0, "clearance": "restricted", "anchor": {"model": "a"}}]
    base = ModelMeanPredictor().fit(np.zeros((4, 3)), np.array([[1.0]] * 4), ["a"])
    eco = EcoRoute(base, "fake", catalog)
    monkeypatch.setattr(eco, "embed", lambda texts: np.zeros((len(texts), 3), np.float32))
    d = eco.route("hello")
    eco.parallel_privacy = True
    assert eco.route("mail jane.doe@example.com").level.name == "CONFIDENTIAL"
    assert {"privacy", "embed", "predict", "decide", "explain", "total"} <= set(d.timings_ms)
    assert "total=" in d.headers()["X-EcoRoute-Time-Ms"]
