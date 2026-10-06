"""Route example prompts and save each decision, with its graph paths, as JSON.

    python scripts/decision_example.py --router artifacts/router.pt --out reports/decisions.json

The file feeds scripts/draw_decision.py, which draws the README's worked example, so every
number in that figure comes from a real route() call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecoroute.gateway.load import load_gateway_router
from ecoroute.routing.decision import difficulty_label
from ecoroute.service import EcoRoute

EXAMPLES = [
    ("Translate 'thank you very much' into French.", None),
    ("What is the capital of Australia?", None),
    ("Write a Python function that merges overlapping intervals, with tests.", None),
    ("Prove that there are infinitely many primes of the form 4k+3.", None),
    ("A train leaves at 3pm at 80 km/h; another at 4pm at 100 km/h. When does it catch up?", None),
    ("Explain the difference between a mutex and a semaphore, with an example.", None),
    ("Summarise our Q3 board deck in five bullet points.", {"sensitivity_label": "Confidential"}),
    ("Email jane.doe@example.com a reminder about Friday's meeting.", None),
]


def decision_record(eco: EcoRoute, prompt: str, context, profile: str) -> dict:
    d = eco.route(prompt, context=context, policy=profile)
    rec = {
        "prompt": prompt,
        "context": context,
        "profile": profile,
        "tau": d.policy.tau,
        "fallback_margin": d.policy.fallback_margin,
        "lambda_energy": d.policy.lambda_energy,
        "model": d.model,
        "level": str(d.level),
        "difficulty": d.difficulty,
        "difficulty_label": None if d.difficulty is None else difficulty_label(d.difficulty),
        "below_floor": d.below_floor,
        "steps": d.steps,
        "headers": d.headers(),
        "candidates": [
            {
                "name": c.name,
                "p_success": c.p_success,
                "cost_usd": c.cost_usd,
                "energy_wh": c.energy_wh,
                "score": c.score(d.policy),
                "allowed": c.allowed,
                "qualifies": c.qualifies,
                "excluded_because": c.excluded_because,
            }
            for c in d.candidates
        ],
    }
    graph = eco.graph()
    if graph is not None:
        x = eco.embed([prompt])[0]
        g = graph.explain(x)
        rec["graph"] = {
            "skills": g.skills,
            "levels": g.levels,
            "beta": g.beta,
            # Path weights per benchmark model: skill, level, solve rate, weight.
            "edges": {m: sorted(e, key=lambda t: -t[3])[:6] for m, e in g.edges.items()},
            "neighbours": g.neighbours,
        }
        rec["anchors"] = {
            name: {"model": a.model, "shift": a.shift, "source": a.source}
            for name, a in eco.predictor.anchors.items()
        }
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/decisions.json"))
    parser.add_argument("--no-pii-model", action="store_true")
    parser.add_argument("--pii-device", type=int, default=None)
    args = parser.parse_args()
    # The gateway's own router (all catalog models, full privacy check), as in e2e_check.py.
    eco = load_gateway_router(
        args.router, args.catalog, {}, pii_model=not args.no_pii_model,
        pii_device=args.pii_device, dry_run=True,
    )  # fmt: skip
    out = [
        decision_record(eco, p, ctx, prof)
        for p, ctx in EXAMPLES
        for prof in ("quality", "balanced", "eco")
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    for r in out:
        print(f"[{r['profile']}] {r['prompt'][:50]} -> {r['model']} ({r['level']})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
