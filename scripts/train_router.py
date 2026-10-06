"""Train the predictor the router deploys and save it.

    python scripts/train_router.py --data data/processed --out artifacts/router.pt

Uses SPROUT (the most recent model pool, which the catalog anchors point at). Trains the
decision graph (prompt -> skills -> difficulty -> models) and, for comparison, an ensemble
of kNN, matrix factorization, IRT and MLP; both are calibrated on validation. Prints both
on test and saves the one named by --predictor (default: the graph).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ecoroute.data.prices import fill_missing_costs
from ecoroute.eval.metrics import held_out_saving, quality_report
from ecoroute.features.embed import DEFAULT_ENCODER, cached_embeddings
from ecoroute.graph import SkillGraph
from ecoroute.predictors import (
    CalibratedPredictor,
    EnsemblePredictor,
    IRTPredictor,
    KNNPredictor,
    MatrixFactorizationPredictor,
    MLPPredictor,
    cost_matrix,
    outcome_matrix,
)
from ecoroute.service import EcoRoute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--source", default="sprout")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--predictor", choices=["graph", "ensemble"], default="graph")
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
        skills[split] = part.task.to_numpy()
    (X_tr, Y_tr, _), (X_va, Y_va, C_va), (X_te, Y_te, C_te) = (
        data["train"],
        data["val"],
        data["test"],
    )

    graph = SkillGraph().fit(X_tr, Y_tr, models, skills=skills["train"])
    print(f"graph skills: {graph.skills_}")
    members = [
        KNNPredictor(k=32),
        MatrixFactorizationPredictor(epochs=args.epochs),
        IRTPredictor(epochs=args.epochs),
        MLPPredictor(epochs=args.epochs),
    ]
    for m in members:
        m.fit(X_tr, Y_tr, models)
    candidates = {
        "graph": CalibratedPredictor.wrap(graph).calibrate(X_va, Y_va),
        "ensemble": CalibratedPredictor.wrap(EnsemblePredictor.of_fitted(members)).calibrate(
            X_va, Y_va
        ),
    }
    for name, pred in candidates.items():
        P_te = pred.predict_proba(X_te)
        q = {k: round(v, 4) for k, v in quality_report(P_te, Y_te).items()}
        h = held_out_saving(pred.predict_proba(X_va), Y_va, C_va, P_te, Y_te, C_te, models)
        print(f"\n[{name}] test quality: {q}")
        if h["matched"]:
            verdict = "within" if h["within_tolerance"] else "OUTSIDE"
            print(
                f"[{name}] tau {h['tau']:.2f} chosen on validation: test accuracy "
                f"{h['router_accuracy']:.4f} vs {h['reference_accuracy']:.4f} "
                f"({verdict} the 1-point tolerance), saving {h['cost_saving_pct']:.1f}%"
            )
        else:
            print(f"[{name}] no threshold kept accuracy within 1 point on validation")
    predictor = candidates[args.predictor]

    catalog = yaml.safe_load(args.catalog.read_text())["models"]
    eco = EcoRoute(predictor, args.encoder, catalog)  # fails here if an anchor is missing
    eco.save(args.out)
    print(f"saved {args.out} ({len(eco.predictor.names)} catalog models anchored)")


if __name__ == "__main__":
    main()
