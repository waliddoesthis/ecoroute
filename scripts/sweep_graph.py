"""Choose the graph's settings on validation, then report the choice once on test.

    python scripts/sweep_graph.py --data data/processed

Each setting (classifier strength C, similar prompts k, shrinkage) is fitted on train;
beta is tuned and the balanced tau chosen on the first half of validation, calibration
fitted on the second half, as in train_router.py. Settings are ranked by the cost of the
balanced tau on the first validation half (lower is better, among those meeting the
1-point target with margin); test is printed for the winner and the current default only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoroute.data.prices import fill_missing_costs
from ecoroute.eval.metrics import evaluate_at_tau, gap_upper_bound, held_out_saving, quality_report
from ecoroute.features.embed import DEFAULT_ENCODER, cached_embeddings
from ecoroute.graph import SkillGraph, skill_name
from ecoroute.predictors import CalibratedPredictor, cost_matrix, outcome_matrix

DEFAULT = {"C": 1.0, "k": 32, "shrink": 20.0}
# C x k at the default shrinkage, plus shrinkage alone at the default C and k.
GRID = [(C, k, 20.0) for C in (0.3, 1.0, 3.0) for k in (16, 32, 64)] + [
    (1.0, 32, 10.0),
    (1.0, 32, 40.0),
]
TAUS = np.round(np.arange(0.50, 1.0, 0.01), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--source", default="sprout")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = parser.parse_args()

    outcomes = fill_missing_costs(pd.read_parquet(args.data / "outcomes.parquet"))
    outcomes = outcomes[outcomes.source == args.source]
    models = sorted(outcomes.model.unique())
    cache = args.data / f"embeddings_{args.encoder.replace('/', '_')}.npz"
    data, skills = {}, {}
    for split in ("train", "val", "test"):
        part = outcomes[outcomes.split == split].drop_duplicates("prompt_id")
        ids = part.prompt_id.tolist()
        X = cached_embeddings(ids, part.prompt.tolist(), cache, encoder=args.encoder)
        data[split] = (X, outcome_matrix(outcomes, ids, models), cost_matrix(outcomes, ids, models))
        skills[split] = part.task.map(skill_name).to_numpy()
    (X_tr, Y_tr, _), (X_va, Y_va, C_va), (X_te, Y_te, C_te) = (
        data["train"],
        data["val"],
        data["test"],
    )
    lab = pd.read_parquet(args.data / "prompts.parquet")
    lab = lab[(lab.split == "train") & lab.difficulty.isin(["easy", "medium", "hard"])]
    X_lab = cached_embeddings(lab.prompt_id.tolist(), lab.prompt.tolist(), cache, args.encoder)
    level_extra = (X_lab, lab.difficulty.map({"easy": 0, "medium": 1, "hard": 2}).to_numpy())

    half = len(X_va) // 2
    A, B = slice(0, half), slice(half, None)
    ref = int(np.nanargmax(np.nanmean(Y_va[A], axis=0)))
    rows, fitted = [], {}
    for C, k, shrink in GRID:
        g = SkillGraph(C=C, k=k, shrink=shrink).fit(
            X_tr, Y_tr, models, skills=skills["train"], level_extra=level_extra
        )
        g.tune_beta(X_va[A], Y_va[A])
        pred = CalibratedPredictor.wrap(g).calibrate(X_va[B], Y_va[B])
        P_a = pred.predict_proba(X_va[A])
        bound = gap_upper_bound(P_a, Y_va[A], C_va[A], ref, TAUS, 1.645)
        ok = TAUS[bound <= 0.01]
        costs = [evaluate_at_tau(P_a, Y_va[A], C_va[A], t)["cost"] for t in ok]
        q = quality_report(P_a, Y_va[A])
        rows.append(
            {"C": C, "k": k, "shrink": shrink, "beta": g.beta, "val_brier": q["brier"],
             "val_auc": q["auc"], "val_cost": min(costs) if costs else np.inf}
        )  # fmt: skip
        fitted[(C, k, shrink)] = pred
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows).sort_values(["val_cost", "val_brier"])
    print("\nsettings ranked on validation (first half):")
    print(table.to_string(index=False, float_format="{:.4f}".format))
    best = table.iloc[0]
    for name, key in [
        ("chosen", (best.C, int(best.k), best.shrink)),
        ("default", (DEFAULT["C"], DEFAULT["k"], DEFAULT["shrink"])),
    ]:
        pred = fitted[key]
        P_te = pred.predict_proba(X_te)
        h = held_out_saving(
            pred.predict_proba(X_va[A]), Y_va[A], C_va[A], P_te, Y_te, C_te, models,
            0.01, TAUS, z=1.645,
        )  # fmt: skip
        q = {k: round(v, 4) for k, v in quality_report(P_te, Y_te).items()}
        print(f"\n[{name} C={key[0]} k={key[1]} shrink={key[2]}] test quality {q}")
        if h["matched"]:
            lo, hi = h["acc_gap_ci90"]
            print(
                f"  balanced tau {h['tau']:.2f}: test gap {100 * h['acc_gap']:.2f} pts "
                f"(90% CI {100 * lo:.2f} to {100 * hi:.2f}), saving {h['cost_saving_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
