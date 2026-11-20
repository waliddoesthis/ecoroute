"""Does retraining on your own graded prompts help? Measured on public data.

    python scripts/eval_adapt.py --data data/processed --router artifacts/router.pt

Setting: your work is one domain (default: SPROUT's MATH prompts from the test split, which
the router never trained on). You deploy two models the router only knows through anchors,
and the anchors are wrong: the small model is anchored to llama-3.1-8b but really behaves
like llama-3.3-70b, the big one is anchored to gpt-4o but behaves like claude-3.5-sonnet.
"Grading" a prompt reveals the true model's real SPROUT outcome.

For n graded prompts per model, adapt() (shifts + similar-prompt evidence) is fitted on n
prompts and checked on a fixed held-out set of the same domain:
- Brier score of the small model's predictions (lower is better);
- routing at the router's balanced quality floor: share of prompts sent to the small model
  and accuracy, against always using the big one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoroute.adapt import adapt
from ecoroute.predictors.local import brier
from ecoroute.service import EcoRoute

PAIRS = {  # deployed name: (anchor the router assumes, model that really answers)
    "small-new": ("llama-3.1-8b-instruct", "llama-3.3-70b-instruct"),
    "big-new": ("gpt-4o", "claude-3.5-sonnet"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--task", default="lighteval/MATH")
    parser.add_argument("--n", type=int, nargs="+", default=[25, 100, 400])
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    o = pd.read_parquet(args.data / "outcomes.parquet")
    o = o[(o.split == "test") & o.task.str.startswith(args.task)]
    truth = o.pivot_table(index="prompt", columns="model", values="correct", aggfunc="mean")
    truth = truth.dropna(subset=[t for _, t in PAIRS.values()])
    prompts = truth.index.to_numpy()
    print(f"{len(prompts)} {args.task} test prompts with outcomes for both true models")

    catalog = [
        {"name": n, "provider": "x", "api_id": n, "clearance": "internal",
         "price_out_per_mtok": 1.0 if n == "small-new" else 15.0, "anchor": {"model": a}}
        for n, (a, _) in PAIRS.items()
    ]  # fmt: skip
    eco = EcoRoute.load(args.router)
    eco = EcoRoute(eco.predictor.base, eco.encoder_name, catalog, taus=eco.taus)
    X = dict(zip(prompts, eco.embed(list(prompts))))
    eco.embed = lambda texts: np.stack([X[t] for t in texts])
    tau = eco.router.profiles["balanced"].tau

    def evaluate(test: np.ndarray) -> dict:
        P = eco.predictor.predict_proba(np.stack([X[t] for t in test]))
        names = eco.predictor.names
        ps, used, correct = [], 0, []
        for t, row in zip(test, P):
            p = dict(zip(names, map(float, row)))
            if eco.local is not None:
                p = eco.local.adjust(X[t], p)
            ps.append(p["small-new"])
            small = p["small-new"] >= tau
            used += small
            true = PAIRS["small-new" if small else "big-new"][1]
            correct.append(truth.loc[t, true])
        y = truth.loc[test, PAIRS["small-new"][1]].to_numpy()
        return {"brier": brier(np.array(ps), y), "small_share": used / len(test),
                "accuracy": float(np.mean(correct))}  # fmt: skip

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(prompts)
        test, pool = order[: args.test], order[args.test :]
        eco.local, eco.shifts = None, {}
        for a in eco.predictor.anchors.values():
            a.shift = 0.0
        before = evaluate(test)
        big_acc = float(truth.loc[test, PAIRS["big-new"][1]].mean())
        small_acc = float(truth.loc[test, PAIRS["small-new"][1]].mean())
        for n in args.n:
            graded = [
                {"prompt": t, "model": name, "success": float(truth.loc[t, true])}
                for t in pool[:n]
                for name, (_, true) in PAIRS.items()
            ]
            adapt(eco, graded, test_share=0.0, min_per_model=min(30, n))
            after = evaluate(test)
            rows.append({"seed": seed, "n": n, "brier_before": before["brier"],
                         "brier_after": after["brier"], "small_before": before["small_share"],
                         "small_after": after["small_share"], "acc_before": before["accuracy"],
                         "acc_after": after["accuracy"], "acc_always_big": big_acc,
                         "acc_always_small": small_acc})  # fmt: skip
    df = pd.DataFrame(rows).groupby("n").mean(numeric_only=True).drop(columns="seed")
    pd.set_option("display.width", 200)
    print(f"balanced tau {tau:.2f}; held-out {args.test} prompts; mean of {args.seeds} seeds")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
