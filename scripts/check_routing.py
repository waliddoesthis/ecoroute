"""Does the trained router send hard prompts to stronger models?

    python scripts/check_routing.py --router artifacts/router.pt --data data/processed

Uses the RouterArena prompts (labelled easy / medium / hard, no outcomes, never seen in
training). Reports, per label: the predicted difficulty, and which catalog tier the router
picks under each policy profile. A useful router shows difficulty rising from easy to hard
and the tier mix shifting up with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecoroute.features.embed import cached_embeddings
from ecoroute.routing import PROFILES
from ecoroute.service import EcoRoute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--n", type=int, default=3000, help="prompts to sample (0 = all)")
    args = parser.parse_args()

    eco = EcoRoute.load(args.router, catalog=args.catalog)
    prompts = pd.read_parquet(args.data / "prompts.parquet")
    prompts = prompts[prompts.difficulty.isin(["easy", "medium", "hard"])]
    if args.n:
        prompts = prompts.sample(min(args.n, len(prompts)), random_state=0)
    cache = args.data / f"embeddings_{eco.encoder_name.replace('/', '_')}.npz"
    X = cached_embeddings(
        prompts.prompt_id.tolist(), prompts.prompt.tolist(), cache, eco.encoder_name
    )

    base = eco.predictor.base.predict_proba(X)
    prompts = prompts.assign(difficulty_pred=1 - base.mean(axis=1))
    print("predicted difficulty (1 - mean P over training models) by RouterArena label:")
    print(prompts.groupby("difficulty").difficulty_pred.describe()[["count", "mean", "50%"]])
    hard_vs_easy = prompts[prompts.difficulty.isin(["easy", "hard"])]
    auc = roc_auc_score(hard_vs_easy.difficulty == "hard", hard_vs_easy.difficulty_pred)
    print(f"AUC, hard vs easy from predicted difficulty: {auc:.3f} (0.5 = no signal)")

    P = eco.predictor.predict_proba(X)
    tier = {m["name"]: m["tier"] for m in eco.catalog}
    print("\ncatalog P(correct), mean by label:")
    print(
        pd.DataFrame(P, columns=eco.predictor.names)
        .groupby(prompts.difficulty.values)
        .mean()
        .round(3)
        .T
    )
    for name in PROFILES:
        picks = [
            eco.router.decide(t, dict(zip(eco.predictor.names, p)), policy=name).model
            for t, p in zip(prompts.prompt, P)
        ]
        mix = pd.crosstab(
            prompts.difficulty.values, np.array([tier[m] for m in picks]), normalize="index"
        )
        print(f"\npolicy {name}: share of prompts per chosen tier")
        print(mix.round(3))


if __name__ == "__main__":
    main()
