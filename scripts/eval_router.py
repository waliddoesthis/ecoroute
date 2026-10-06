"""Reproduce the README results table from a saved router, and save the report.

    python scripts/eval_router.py --router artifacts/router_graph_v14.pt --data data/processed

Uses the thresholds and fallback margins stored in the router file (chosen on validation
when it was trained) and evaluates them once on the SPROUT test split. Writes
reports/eval_<router>.json with every number, the test-set size, the git commit and the
pricing source, and prints the table.

- Test split: prompts whose id hash falls in the test bucket (src/ecoroute/data/schema.py
  assign_split, 10%); only prompts graded for every model are used, so all rows compare
  the same prompts.
- Accuracy change: router accuracy minus the reference's, in percentage points (positive
  means the router is more accurate). The reference is the most accurate single model on
  validation (GPT-4o).
- 90% CI: percentile bootstrap over test prompts (1,000 resamples, paired, seed 0).
- Cost: SPROUT token counts priced with the per-token prices its authors published
  (src/ecoroute/data/prices.py). The router's own running cost is not included.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ecoroute.data.prices import fill_missing_costs
from ecoroute.eval.metrics import bootstrap_gap, evaluate_at_tau
from ecoroute.features.embed import cached_embeddings
from ecoroute.predictors import cost_matrix, outcome_matrix
from ecoroute.routing import PROFILE_TOLERANCE
from ecoroute.service import EcoRoute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--source", default="sprout")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    eco = EcoRoute.load(args.router, catalog=args.catalog)
    pred = eco.predictor.base
    models = list(pred.models)
    outcomes = fill_missing_costs(pd.read_parquet(args.data / "outcomes.parquet"))
    outcomes = outcomes[outcomes.model.isin(models)]
    test = outcomes[(outcomes.split == "test") & (outcomes.source == args.source)]
    part = test.drop_duplicates("prompt_id")
    ids = part.prompt_id.tolist()
    Y, C = outcome_matrix(test, ids, models), cost_matrix(test, ids, models)
    full = ~np.isnan(Y).any(axis=1) & ~np.isnan(C).any(axis=1)
    cols = ~np.isnan(Y[full]).all(axis=0)
    if not full.any():  # sources graded on a subset of models: keep the models they cover
        cols = ~np.isnan(Y).all(axis=0)
        full = ~np.isnan(Y[:, cols]).any(axis=1) & ~np.isnan(C[:, cols]).any(axis=1)
    cache = args.data / f"embeddings_{eco.encoder_name.replace('/', '_')}.npz"
    sel = part[full]
    X = cached_embeddings(sel.prompt_id.tolist(), sel.prompt.tolist(), cache, eco.encoder_name)
    P = pred.predict_proba(X)[:, cols]
    Y, C = Y[full][:, cols], C[full][:, cols]
    names = [m for m, k in zip(models, cols) if k]
    ref = names.index("gpt-4o") if "gpt-4o" in names else int(np.argmax(Y.mean(axis=0)))
    ref_acc, ref_cost = float(Y[:, ref].mean()), float(C[:, ref].mean())

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    rows = []
    for name in PROFILE_TOLERANCE:
        pol = eco.router.profiles[name]
        got = evaluate_at_tau(P, Y, C, pol.tau, pol.fallback_margin)
        lo, hi = bootstrap_gap(P, Y, C, ref, pol.tau, margin=pol.fallback_margin)
        rows.append(
            {
                "profile": name,
                "target_max_loss_pp": 100 * PROFILE_TOLERANCE[name],
                "tau": pol.tau,
                "fallback_margin": pol.fallback_margin,
                "accuracy": got["accuracy"],
                # Positive: the router is more accurate than the reference.
                "accuracy_change_pp": 100 * (got["accuracy"] - ref_acc),
                "accuracy_change_ci90_pp": [-100 * hi, -100 * lo],
                "cost_per_request_usd": got["cost"],
                "cost_saved_pct": 100 * (1 - got["cost"] / ref_cost),
            }
        )
    report = {
        "router": str(args.router),
        "commit": commit,
        "source": args.source,
        "split": "test (prompt-id hash bucket, 10%)",
        "n_test_prompts": int(len(Y)),
        "models": names,
        "reference": names[ref],
        "reference_accuracy": ref_acc,
        "reference_cost_per_request_usd": ref_cost,
        "ci": "90% percentile bootstrap over test prompts, 1,000 paired resamples, seed 0",
        "pricing": "SPROUT token counts x prices published with SPROUT (data/prices.py)",
        "router_overhead_included": False,
        "profiles": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"eval_{args.router.stem}_{args.source}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"{args.router} at {commit}: {len(Y)} {args.source} test prompts, reference "
          f"{names[ref]} (accuracy {ref_acc:.4f}, ${ref_cost:.6f} per request)\n")  # fmt: skip
    print("| Profile | Target | Accuracy change vs reference, pp (90% CI) | Cost saved |")
    print("|---|---|---|---|")
    for r in rows:
        lo, hi = r["accuracy_change_ci90_pp"]
        target = "no loss" if r["target_max_loss_pp"] == 0 else f"≥ -{r['target_max_loss_pp']:.0f}"
        print(f"| {r['profile']} | {target} | {r['accuracy_change_pp']:+.2f} ({lo:+.2f} to "
              f"{hi:+.2f}) | {r['cost_saved_pct']:.1f}% |")  # fmt: skip
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
