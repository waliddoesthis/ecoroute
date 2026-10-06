import pytest

from ecoroute.confidentiality import Level
from ecoroute.routing import PROFILES, NoAllowedModel, Router

AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"


@pytest.fixture
def router():
    return Router.from_yaml("configs/models.yaml", policy="balanced")


def test_easy_prompt_goes_to_cheapest_qualifying_external_model(router):
    p = {m["name"]: 0.9 for m in router.catalog}
    d = router.decide("Translate 'good morning' into Spanish", p)
    assert d.level == Level.INTERNAL
    allowed = [c for c in d.candidates if c.allowed]
    assert d.chosen.cost_usd == min(c.cost_usd for c in allowed if c.qualifies)
    assert "cheaper than" in d.explain()


def test_hard_prompt_goes_to_premium(router):
    p = {m["name"]: 0.2 for m in router.catalog}
    p["claude-opus-5-5"] = 0.9
    d = router.decide("Prove the Riemann hypothesis for this special case ...", p)
    assert d.model == "claude-opus-5-5"
    assert d.difficulty > 0.6 and "hard" in d.headers()["X-EcoRoute-Difficulty"]


def test_restricted_prompt_never_leaves_the_local_model(router):
    p = {m["name"]: 0.99 for m in router.catalog}
    p["llama-3.1-8b-local"] = 0.1  # even when the local model is predicted to fail
    d = router.decide(f"why is {AWS} rejected?", p)
    assert d.level == Level.RESTRICTED
    assert d.model == "llama-3.1-8b-local" and d.below_floor
    assert "aws_access_key" in d.explain() and AWS not in d.explain()
    for c in d.candidates:
        if c.name != d.model:
            assert not c.allowed and "clearance" in c.excluded_because


def test_no_cleared_model_fails_closed():
    r = Router([{"name": "ext", "clearance": "internal", "tier": 1}])
    with pytest.raises(NoAllowedModel):
        r.decide("x", {"ext": 0.9}, context={"sensitivity_label": "restricted"})


def test_profile_changes_the_floor(router):
    p = {m["name"]: 0.5 + 0.1 * m["tier"] for m in router.catalog}
    eco = router.decide("summarise this", p, policy="eco")
    quality = router.decide("summarise this", p, policy="quality")
    assert eco.chosen.cost_usd <= quality.chosen.cost_usd
    assert quality.chosen.p_success >= 0.85


def test_headers_and_unknown_energy_flagged(router):
    p = {m["name"]: 0.95 for m in router.catalog}
    d = router.decide("hello", p, context={"min_level": "restricted"})
    h = d.headers()
    assert h["X-EcoRoute-Model"] == "llama-3.1-8b-local"
    assert h["X-EcoRoute-Level"] == "restricted"
    assert d.chosen.energy_estimated


def test_disabled_models_are_never_chosen(router):
    assert "qwen3.6-35b-a3b-local" not in {m["name"] for m in router.catalog}
    p = {m["name"]: 0.95 for m in router.catalog}
    d = router.decide("hi", p, context={"min_level": "restricted"})
    assert d.model == "llama-3.1-8b-local"


def test_self_hosted_models_are_not_free(router):
    local = next(m for m in router.catalog if m["name"] == "llama-3.1-8b-local")
    cost, wh, estimated = router.true_cost(local, in_tokens=100, out_tokens=500)
    # 500 tokens at 50 tok/s = 10 s at 215 W.
    assert wh == pytest.approx(215 * 10 / 3600)
    assert cost == pytest.approx(wh / 1000 * 0.15) and cost > 0 and estimated


def test_hardware_cost_and_measured_energy():
    r = Router([])
    host = {"hosting": {"gpu_watts": 300, "tokens_per_second": 100, "gpu_usd_per_hour": 3.6}}
    cost, wh, _ = r.true_cost(host, 0, 1000)  # 10 s
    assert cost == pytest.approx(300 * 10 / 3600 / 1000 * 0.15 + 0.01)
    measured = {**host, "energy": {"wh_per_1k_out": 0.2, "source": "measured"}}
    assert r.true_cost(measured, 0, 1000)[1:] == (pytest.approx(0.2), False)


def test_eco_profile_trades_cost_for_energy():
    # A local GPU that is cheap but power-hungry vs an efficient but pricier API model.
    catalog = [
        {"name": "local", "clearance": "restricted",
         "hosting": {"gpu_watts": 215, "tokens_per_second": 50}},
        {"name": "api", "clearance": "internal", "price_out_per_mtok": 0.5,
         "energy": {"wh_per_1k_out": 0.05, "source": "prior"}},
    ]  # fmt: skip
    r = Router(catalog)
    p = {"local": 0.9, "api": 0.9}
    assert r.decide("hi", p, policy="quality").model == "local"  # cheapest in dollars
    assert r.decide("hi", p, policy="eco").model == "api"  # far less energy
    assert r.decide("hi", p, policy="eco", context={"min_level": "restricted"}).model == "local"


def test_tuned_profiles_replace_taus_and_stay_ordered():
    from ecoroute.routing import tuned_profiles

    prof = tuned_profiles({"eco": 0.8, "balanced": 0.9})
    assert prof["eco"].tau == 0.8 and prof["balanced"].tau == 0.9
    # quality was not measured: its default 0.85 would be looser than balanced
    assert prof["quality"].tau == 0.9
    assert prof["eco"].lambda_energy == PROFILES["eco"].lambda_energy
    assert tuned_profiles(None)["balanced"] == PROFILES["balanced"]


def test_router_uses_its_own_profiles():
    from ecoroute.routing import tuned_profiles

    r = Router.from_yaml("configs/models.yaml", profiles=tuned_profiles({"balanced": 0.95}))
    assert r.policy.tau == 0.95
    assert Router.from_yaml("configs/models.yaml").policy.tau == PROFILES["balanced"].tau
