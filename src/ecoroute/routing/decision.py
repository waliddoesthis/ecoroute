"""The routing decision and its explanation (Graph A in the design doc).

The decision is a short path through fixed steps, and each step's outcome is kept so the
explanation is the decision itself, not a story told afterwards:

1. Confidentiality: classify the prompt; drop models without enough clearance (hard rule).
2. Quality floor: keep allowed models whose predicted P(correct) >= tau.
3. Cost and energy: among those, take the lowest cost + lambda_energy * energy.
   If none reaches tau, take the allowed model most likely to succeed.

P(correct) per model comes from a predictor; this module does not care which one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ecoroute.confidentiality import Classification, Detector, Level


class NoAllowedModel(RuntimeError):
    """No model in the catalog is cleared for this prompt; the request must not be sent."""


@dataclass(frozen=True)
class Policy:
    tau: float  # quality floor on predicted P(correct)
    lambda_energy: float  # USD charged per Wh when ranking, to favour efficient models

    @classmethod
    def parse(cls, value: str | Policy, profiles: Mapping[str, Policy] | None = None) -> Policy:
        if isinstance(value, Policy):
            return value
        profiles = PROFILES if profiles is None else profiles
        try:
            return profiles[value]
        except KeyError:
            raise ValueError(f"unknown policy profile {value!r}; use {sorted(profiles)}") from None


# These taus are fallbacks: a trained router stores the tau that each profile's accuracy
# target needs on held-out data (see tuned_profiles), and those replace them.
# lambda_energy prices energy in the ranking. balanced uses about $0.25 per kWh, roughly
# electricity plus the social cost of its carbon (about $0.2/kg CO2 at 0.4 kg/kWh); eco
# weighs energy ten times more; quality ignores it.
PROFILES = {
    "eco": Policy(tau=0.6, lambda_energy=0.0025),
    "balanced": Policy(tau=0.7, lambda_energy=0.00025),
    "quality": Policy(tau=0.85, lambda_energy=0.0),
}


# Accuracy each profile may give up against always using the most accurate model.
PROFILE_TOLERANCE = {"eco": 0.03, "balanced": 0.01, "quality": 0.0}


def tuned_profiles(taus: Mapping[str, float] | None) -> dict[str, Policy]:
    """PROFILES with tau replaced by the values a trained router measured for them.

    A stricter profile never gets a lower floor than a looser one (eco <= balanced <=
    quality), even when only some taus were measured.
    """
    taus = taus or {}
    out, floor = {}, 0.0
    for name in PROFILE_TOLERANCE:
        floor = max(floor, taus.get(name, PROFILES[name].tau))
        out[name] = replace(PROFILES[name], tau=floor)
    return out


@dataclass
class Candidate:
    name: str
    p_success: float | None
    cost_usd: float
    energy_wh: float
    energy_estimated: bool
    allowed: bool
    qualifies: bool = False
    excluded_because: str | None = None

    def score(self, policy: Policy) -> float:
        return self.cost_usd + policy.lambda_energy * self.energy_wh


@dataclass
class Decision:
    model: str
    level: Level
    difficulty: float | None
    policy: Policy
    candidates: list[Candidate]
    confidentiality: Classification
    steps: list[str] = field(default_factory=list)
    below_floor: bool = False

    @property
    def chosen(self) -> Candidate:
        return next(c for c in self.candidates if c.name == self.model)

    def explain(self) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))

    def headers(self) -> dict[str, str]:
        """Response headers the gateway attaches (see docs, section 2c)."""
        h = {
            "X-EcoRoute-Model": self.model,
            "X-EcoRoute-Level": str(self.level),
            "X-EcoRoute-Reason": self.steps[-1] if self.steps else "",
        }
        if self.difficulty is not None:
            h["X-EcoRoute-Difficulty"] = (
                f"{self.difficulty:.2f} ({difficulty_label(self.difficulty)})"
            )
        return h


def difficulty_label(d: float) -> str:
    return "easy" if d < 0.3 else "hard" if d > 0.6 else "medium"


class Router:
    def __init__(
        self,
        catalog: list[Mapping[str, Any]],
        detector: Detector | None = None,
        policy: str | Policy = "balanced",
        default_wh_per_1k_out: float = 0.3,
        usd_per_kwh: float = 0.15,
        profiles: Mapping[str, Policy] | None = None,
    ) -> None:
        self.catalog = [m for m in catalog if m.get("enabled", True)]
        self.detector = detector or Detector()
        self.profiles = dict(PROFILES if profiles is None else profiles)
        self.policy = Policy.parse(policy, self.profiles)
        # Used when a model's energy is not known yet; such numbers are flagged as estimates.
        self.default_wh_per_1k_out = default_wh_per_1k_out
        # Electricity price for self-hosted models (an API's price already includes it).
        self.usd_per_kwh = usd_per_kwh

    @classmethod
    def from_yaml(cls, path: str | Path, **kw) -> Router:
        return cls(yaml.safe_load(Path(path).read_text())["models"], **kw)

    def decide(
        self,
        prompt: str,
        p_success: Mapping[str, float],
        context: Mapping | None = None,
        in_tokens: int | None = None,
        out_tokens: int = 500,
        policy: str | Policy | None = None,
        scan_text: str | None = None,
    ) -> Decision:
        """Choose a model for `prompt` given predicted P(correct) per catalog model.

        scan_text is what the confidentiality check reads when it differs from the prompt
        (for a chat, the whole conversation rather than the last question). Models missing
        from p_success can still be chosen as a last resort but never qualify on quality.
        Raises NoAllowedModel when nothing is cleared for the prompt.
        """
        pol = Policy.parse(policy, self.profiles) if policy is not None else self.policy
        text = prompt if scan_text is None else scan_text
        if in_tokens is None:
            in_tokens = max(1, len(text) // 4)  # rough: about 4 characters per token
        conf = self.detector.classify(text, context)
        cands = [
            self._candidate(m, conf.level, p_success, in_tokens, out_tokens) for m in self.catalog
        ]
        allowed = [c for c in cands if c.allowed]
        steps = [_confidentiality_step(conf, len(allowed), len(cands))]
        if not allowed:
            raise NoAllowedModel(steps[0])

        for c in allowed:
            c.qualifies = c.p_success is not None and c.p_success >= pol.tau
            if not c.qualifies:
                c.excluded_because = (
                    "no quality prediction"
                    if c.p_success is None
                    else f"P(correct) {c.p_success:.2f} < tau {pol.tau:.2f}"
                )
        qualified = [c for c in allowed if c.qualifies]
        steps.append(
            f"Quality floor tau={pol.tau:.2f}: {len(qualified)} of {len(allowed)} allowed "
            f"models are predicted to succeed."
        )

        if qualified:
            best = min(qualified, key=lambda c: (c.score(pol), -(c.p_success or 0)))
            steps.append(_choice_step(best, allowed))
        else:
            best = max(allowed, key=lambda c: (c.p_success or -1, -c.score(pol)))
            steps.append(
                f"No allowed model reaches tau, so {best.name} was chosen as the most likely "
                f"to succeed (P={_fmt_p(best.p_success)}, {_fmt_cost(best.cost_usd)})."
            )

        known = [p for p in p_success.values() if p is not None]
        difficulty = 1 - sum(known) / len(known) if known else None
        return Decision(
            model=best.name,
            level=conf.level,
            difficulty=difficulty,
            policy=pol,
            candidates=cands,
            confidentiality=conf,
            steps=steps,
            below_floor=not qualified,
        )

    def _candidate(self, m, level, p_success, in_tokens, out_tokens) -> Candidate:
        cost, energy, estimated = self.true_cost(m, in_tokens, out_tokens)
        clearance = Level.parse(m.get("clearance", "public"))
        allowed = clearance >= level
        return Candidate(
            name=m["name"],
            p_success=p_success.get(m["name"]),
            cost_usd=cost,
            energy_wh=energy,
            energy_estimated=estimated,
            allowed=allowed,
            excluded_because=None if allowed else f"clearance {clearance} < prompt level {level}",
        )

    def true_cost(self, m: Mapping[str, Any], in_tokens: int, out_tokens: int):
        """(USD, Wh, energy_is_estimate) for one answer from model m.

        No model is free. An API model costs its token price. A self-hosted model (one with
        a `hosting` block) costs the electricity its GPU draws while generating, plus the
        hardware's hourly cost spread over that time:
            seconds = out_tokens / tokens_per_second
            Wh      = gpu_watts * seconds / 3600
            USD     = Wh / 1000 * usd_per_kwh + gpu_usd_per_hour * seconds / 3600
        Measured Wh per 1K tokens, when present, replaces the wattage estimate.
        """
        price = (
            in_tokens * m.get("price_in_per_mtok", 0.0)
            + out_tokens * m.get("price_out_per_mtok", 0.0)
        ) / 1e6
        energy_cfg = m.get("energy") or {}
        wh_per_1k = energy_cfg.get("wh_per_1k_out")
        measured = wh_per_1k is not None and energy_cfg.get("source") == "measured"
        host = m.get("hosting")
        if host:
            seconds = out_tokens / float(host["tokens_per_second"])
            wh = wh_per_1k * out_tokens / 1000 if measured else host["gpu_watts"] * seconds / 3600
            cost = (
                price
                + wh / 1000 * self.usd_per_kwh
                + host.get("gpu_usd_per_hour", 0.0) * seconds / 3600
            )
            return cost, wh, not measured
        wh = (
            (wh_per_1k if wh_per_1k is not None else self.default_wh_per_1k_out) * out_tokens / 1000
        )
        return price, wh, not measured


def _fmt_p(p: float | None) -> str:
    return "unknown" if p is None else f"{p:.2f}"


def _fmt_cost(usd: float) -> str:
    return f"${usd:.5f}"


def _confidentiality_step(conf: Classification, n_allowed: int, n_total: int) -> str:
    why = "; ".join(conf.reasons()[:3]) or "no sensitive content found"
    return f"Confidentiality: {conf.level} ({why}). {n_allowed} of {n_total} models are cleared."


def _choice_step(best: Candidate, allowed: list[Candidate]) -> str:
    est = " (estimated)" if best.energy_estimated else ""
    text = (
        f"Chose {best.name}: cheapest qualifying model "
        f"(P={_fmt_p(best.p_success)}, {_fmt_cost(best.cost_usd)}, {best.energy_wh:.3f} Wh{est})."
    )
    priciest = max(allowed, key=lambda c: c.cost_usd)
    if priciest is not best and priciest.cost_usd > 0:
        saved = 100 * (1 - best.cost_usd / priciest.cost_usd)
        text += f" {saved:.0f}% cheaper than the priciest allowed model, {priciest.name}."
    return text
