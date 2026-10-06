"""Does the trained router send hard prompts to stronger models?

    python scripts/check_routing.py --router artifacts/router.pt --data data/processed

Uses the RouterArena prompts (labelled easy / medium / hard, no outcomes, never seen in
training). Reports, per label: the predicted difficulty, and which catalog tier the router
picks under each policy profile. A useful router shows difficulty rising from easy to hard
and the tier mix shifting up with it.

It also reports, per profile, the mean price and energy per request of the chosen catalog
models against sending every prompt to the most capable catalog model. Prices are list
prices and energy mostly tier priors (see configs/models.yaml), so these are estimates;
docs/impact.md scales them to company volumes.
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
    # Only val/test prompts: the graph may have learned difficulty from the train split.
    prompts = prompts[
        prompts.difficulty.isin(["easy", "medium", "hard"]) & (prompts.split != "train")
    ]
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
    # The reference: the highest-tier catalog model (the priciest one on a tie).
    top = max(eco.catalog, key=lambda m: (m["tier"], m.get("price_out_per_mtok") or 0))["name"]
    usage = []
    for name in PROFILES:
        decisions = [
            eco.router.decide(t, dict(zip(eco.predictor.names, p)), policy=name)
            for t, p in zip(prompts.prompt, P)
        ]
        picks = [d.model for d in decisions]
        mix = pd.crosstab(
            prompts.difficulty.values, np.array([tier[m] for m in picks]), normalize="index"
        )
        print(f"\npolicy {name}: share of prompts per chosen tier")
        print(mix.round(3))
        ref = [next(c for c in d.candidates if c.name == top) for d in decisions]
        cost = np.mean([d.chosen.cost_usd for d in decisions])
        energy = np.mean([d.chosen.energy_wh for d in decisions])
        ref_cost = np.mean([c.cost_usd for c in ref])
        ref_energy = np.mean([c.energy_wh for c in ref])
        usage.append(
            {"policy": name, "usd_per_request": cost, "wh_per_request": energy,
             f"usd_always_{top}": ref_cost, f"wh_always_{top}": ref_energy,
             "cost_saved_pct": 100 * (1 - cost / ref_cost),
             "energy_saved_pct": 100 * (1 - energy / ref_energy),
             "local_share": float(np.mean([tier[m] == 0 for m in picks]))}
        )  # fmt: skip
        shares = pd.Series(picks).value_counts(normalize=True).round(3)
        print(
            f"policy {name}: share per model: " + ", ".join(f"{k} {v}" for k, v in shares.items())
        )
    print(f"\nper request, against always using {top} (estimates: list prices, energy priors):")
    print(pd.DataFrame(usage).to_string(index=False, float_format="{:.5g}".format))


if __name__ == "__main__":
    main()
