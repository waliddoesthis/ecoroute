"""Train and compare the baseline predictors on the unified outcomes table.

    python scripts/train_baselines.py --data data/processed --source routerbench

Each source is evaluated on its own, because the sources share few models. Writes
reports/baselines_<source>.json and prints a comparison table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ecoroute.data.prices import fill_missing_costs
from ecoroute.eval.metrics import (
    breakdown_by_difficulty,
    quality_report,
    routing_curve,
    savings_at_quality,
)
from ecoroute.features.embed import DEFAULT_ENCODER, cached_embeddings
from ecoroute.predictors import (
    CalibratedPredictor,
    EnsemblePredictor,
    IRTPredictor,
    KNNPredictor,
    MatrixFactorizationPredictor,
    ModelMeanPredictor,
    cost_matrix,
    outcome_matrix,
)


def split_frame(outcomes: pd.DataFrame, split: str) -> tuple[list[str], list[str]]:
    part = outcomes[outcomes.split == split].drop_duplicates("prompt_id")
    return part.prompt_id.tolist(), part.prompt.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--source", default="routerbench")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    outcomes = pd.read_parquet(args.data / "outcomes.parquet")
    # Tables built before SPROUT costs existed get them filled from token counts here.
    outcomes = fill_missing_costs(outcomes[outcomes.source == args.source])
    models = sorted(outcomes.model.unique())
    cache = args.data / f"embeddings_{args.encoder.replace('/', '_')}.npz"

    data = {}
    for split in ("train", "val", "test"):
        ids, texts = split_frame(outcomes, split)
        X = cached_embeddings(ids, texts, cache, encoder=args.encoder)
        data[split] = (X, outcome_matrix(outcomes, ids, models), cost_matrix(outcomes, ids, models))
    X_tr, Y_tr, _ = data["train"]
    X_va, Y_va, _ = data["val"]
    X_te, Y_te, C_te = data["test"]
    print(
        f"{args.source}: {len(models)} models, {len(X_tr):,} train / {len(X_va):,} val / {len(X_te):,} test prompts"
    )

    predictors = [
        ModelMeanPredictor(),
        KNNPredictor(k=32),
        MatrixFactorizationPredictor(epochs=args.epochs),
        IRTPredictor(epochs=args.epochs),
    ]
    results = []

    def evaluate(pred, seconds: float) -> None:
        P = pred.predict_proba(X_te)
        row = {"predictor": pred.name, **quality_report(P, Y_te), "train_s": seconds}
        curve = routing_curve(P, Y_te, C_te, models)
        if curve.attrs["n_prompts"] > 0:
            singles = curve[curve.policy.str.startswith("always")]
            best_single = singles.loc[singles.accuracy.idxmax()].policy
            saving = savings_at_quality(curve, best_single)
            row["savings_vs_best_model"] = saving
            row["curve"] = curve.to_dict("records")
            if saving["matched"]:
                tau = float(saving["router_policy"].split("=")[1])
                row["by_difficulty"] = breakdown_by_difficulty(P, Y_te, C_te, tau).to_dict(
                    "records"
                )
        results.append(row)

    for pred in predictors:
        t0 = time.time()
        pred.fit(X_tr, Y_tr, models)
        evaluate(pred, time.time() - t0)
        if not isinstance(pred, ModelMeanPredictor):
            t0 = time.time()
            evaluate(CalibratedPredictor.wrap(pred).calibrate(X_va, Y_va), time.time() - t0)

    learned = [p for p in predictors if not isinstance(p, ModelMeanPredictor)]
    ensemble = EnsemblePredictor.of_fitted(learned)
    evaluate(ensemble, 0.0)
    evaluate(CalibratedPredictor.wrap(ensemble).calibrate(X_va, Y_va), 0.0)

    table = pd.DataFrame(results)[["predictor", "brier", "ece", "auc", "train_s"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for r in results:
        if "savings_vs_best_model" in r:
            print(r["predictor"], r["savings_vs_best_model"])
        if "by_difficulty" in r:
            print(
                pd.DataFrame(r["by_difficulty"]).to_string(
                    index=False, float_format="{:.4f}".format
                )
            )
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": args.source, "encoder": args.encoder, "models": models, "results": results}
    (args.out / f"baselines_{args.source}.json").write_text(
        json.dumps(report, indent=2, default=float)
    )


if __name__ == "__main__":
    main()
