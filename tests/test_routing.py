import pytest

from ecoroute.confidentiality import Level
from ecoroute.routing import NoAllowedModel, Router

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
