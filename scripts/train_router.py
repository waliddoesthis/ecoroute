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

import numpy as np
import pandas as pd
import yaml

from ecoroute.data.prices import fill_missing_costs
from ecoroute.eval.metrics import evaluate_at_tau, held_out_saving, quality_report
from ecoroute.features.embed import DEFAULT_ENCODER, cached_embeddings
from ecoroute.graph import SkillGraph, skill_name
from ecoroute.routing import PROFILE_TOLERANCE
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
    parser.add_argument(
        "--source",
        nargs="+",
        default=["sprout", "bigcodebench"],
        help="outcome sources to train on (bigcodebench adds coding prompts)",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--predictor", choices=["graph", "ensemble"], default="graph")
    parser.add_argument(
        "--no-difficulty-labels",
        action="store_true",
        help="don't add RouterArena's labelled train prompts to the graph's difficulty edge",
    )
    args = parser.parse_args()

    outcomes = fill_missing_costs(pd.read_parquet(args.data / "outcomes.parquet"))
    outcomes = outcomes[outcomes.source.isin(args.source)]
    print("training sources: " + outcomes.source.value_counts().to_string().replace("\n", ", "))
    models = sorted(outcomes.model.unique())
    cache = args.data / f"embeddings_{args.encoder.replace('/', '_')}.npz"
    data, skills, sources = {}, {}, {}
    for split in ("train", "val", "test"):
        part = outcomes[outcomes.split == split].drop_duplicates("prompt_id")
        ids = part.prompt_id.tolist()
        X = cached_embeddings(ids, part.prompt.tolist(), cache, encoder=args.encoder)
        data[split] = (X, outcome_matrix(outcomes, ids, models), cost_matrix(outcomes, ids, models))
        skills[split] = part.task.map(skill_name).to_numpy()
        sources[split] = part.source.to_numpy()
    (X_tr, Y_tr, _), (X_va, Y_va, C_va), (X_te, Y_te, C_te) = (
        data["train"],
        data["val"],
        data["test"],
    )

    level_extra = None
    labelled = args.data / "prompts.parquet"
    if not args.no_difficulty_labels and labelled.exists():
        lab = pd.read_parquet(labelled)
        lab = lab[(lab.split == "train") & lab.difficulty.isin(["easy", "medium", "hard"])]
        X_lab = cached_embeddings(lab.prompt_id.tolist(), lab.prompt.tolist(), cache, args.encoder)
        level_extra = (X_lab, lab.difficulty.map({"easy": 0, "medium": 1, "hard": 2}).to_numpy())
        print(f"difficulty edge: adding {len(lab):,} labelled RouterArena train prompts")
    graph = SkillGraph().fit(X_tr, Y_tr, models, skills=skills["train"], level_extra=level_extra)
    print(f"graph skills: {graph.skills_}")
    # Half of validation sets the similar-prompt edge weight, the other half calibrates.
    half = len(X_va) // 2
    print(f"similar-prompt edge weight beta = {graph.tune_beta(X_va[:half], Y_va[:half]):.1f}")
    members = [
        KNNPredictor(k=32),
        MatrixFactorizationPredictor(epochs=args.epochs),
        IRTPredictor(epochs=args.epochs),
        MLPPredictor(epochs=args.epochs),
    ]
    for m in members:
        m.fit(X_tr, Y_tr, models)
    candidates = {
        "graph": CalibratedPredictor.wrap(graph).calibrate(X_va[half:], Y_va[half:]),
        # Same protocol for both: calibrate on the second half of validation, choose
        # taus on the first, measure on test.
        "ensemble": CalibratedPredictor.wrap(EnsemblePredictor.of_fitted(members)).calibrate(
            X_va[half:], Y_va[half:]
        ),
    }
    for name, pred in candidates.items():
        P_te = pred.predict_proba(X_te)
        q = {k: round(v, 4) for k, v in quality_report(P_te, Y_te).items()}
        h = held_out_saving(
            pred.predict_proba(X_va[:half]), Y_va[:half], C_va[:half], P_te, Y_te, C_te,
            models, 0.01, np.round(np.arange(0.50, 1.0, 0.01), 2), z=1.645,
        )  # fmt: skip
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

    # Each profile's quality floor is the cheapest tau that kept accuracy within its
    # tolerance of the best single model on validation; test shows whether it held.
    taus = {}
    # Taus are chosen on the half of validation the calibrator never saw: on the half it
    # was fitted to, the predictions look better than they are, and the chosen taus
    # missed their targets on test by more than chance would explain (runs 28 and 29).
    tune = slice(0, half)
    P_va, P_te = predictor.predict_proba(X_va[tune]), predictor.predict_proba(X_te)
    Y_tune, C_tune = Y_va[tune], C_va[tune]
    z = 1.645
    print(f"\nprofile taus for {args.predictor} (margin z={z}):")
    grid = np.round(np.arange(0.50, 1.0, 0.01), 2)  # finer and higher than the 0.05 grid
    for profile, tol in PROFILE_TOLERANCE.items():
        # quality takes the closest tau when nothing fully matches the best model. With a
        # margin of 1.645 paired standard errors, a tau qualifies only if validation says,
        # with 95% one-sided confidence, that it meets the profile's target. One standard
        # error (run 28) met the targets on average, but the 90% test intervals ran past
        # them (balanced 0.22 to 1.72 points against a 1-point target).
        h = held_out_saving(
            P_va, Y_tune, C_tune, P_te, Y_te, C_te, models, tol, grid,
            closest=profile == "quality", z=z,
        )  # fmt: skip
        if not h["matched"]:
            print(f"  {profile}: nothing within {100 * tol:.0f} points, keeping the default")
            continue
        taus[profile] = h["tau"]
        print(
            f"  {profile} (within {100 * tol:.0f} pts): tau {h['tau']:.2f}, test accuracy "
            f"{h['router_accuracy']:.4f} vs {h['reference_accuracy']:.4f}, "
            f"gap {100 * h['acc_gap']:.2f} pts (90% CI {100 * h['acc_gap_ci90'][0]:.2f} to "
            f"{100 * h['acc_gap_ci90'][1]:.2f}), saving {h['cost_saving_pct']:.1f}%; "
            f"validation ({h['n_val']} prompts) gap {100 * h['val_gap']:.2f}, "
            f"bound {100 * h['val_gap_bound']:.2f}"
        )

    # Coding prompts lack outcomes for a few models, so the all-models comparisons above
    # skip them. Measure them on their own, among the models BigCodeBench covers.
    code = sources["test"] == "bigcodebench"
    if code.any():
        Yc, Cc, Pc = Y_te[code], C_te[code], P_te[code]
        seen = ~np.isnan(Yc).any(axis=0)
        cols = [m for m, s_ in zip(models, seen) if s_]
        Yc, Cc, Pc = Yc[:, seen], Cc[:, seen], Pc[:, seen]
        q = {k: round(v, 4) for k, v in quality_report(Pc, Yc).items()}
        best = int(np.argmax(Yc.mean(axis=0)))
        print(f"\ncoding test prompts: {int(code.sum())}, models with results: {len(cols)}")
        print(f"  quality on coding prompts: {q}")
        print(f"  best single model: {cols[best]} accuracy {Yc[:, best].mean():.4f}")
        for profile, tau in taus.items():
            got = evaluate_at_tau(Pc, Yc, Cc, tau)
            print(
                f"  {profile} (tau {tau:.2f}): accuracy {got['accuracy']:.4f}, saving "
                f"{100 * (1 - got['cost'] / Cc[:, best].mean()):.1f}% vs {cols[best]}"
            )

    catalog = yaml.safe_load(args.catalog.read_text())["models"]
    eco = EcoRoute(predictor, args.encoder, catalog, taus=taus)  # fails if an anchor is missing
    eco.save(args.out)
    print(f"saved {args.out} ({len(eco.predictor.names)} catalog models anchored)")


if __name__ == "__main__":
    main()
