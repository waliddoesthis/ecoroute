"""Download the public routing datasets and write the unified tables.

    python scripts/build_dataset.py --out data/processed

Writes outcomes.parquet (prompt x model graded results) and prompts.parquet
(prompts without outcomes), then prints a summary. Needs access to huggingface.co,
so run it on Lightning AI or your own machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoroute.data import routerarena, routerbench, sprout


def summarise(outcomes: pd.DataFrame, prompts: pd.DataFrame) -> str:
    lines = [
        f"outcomes: {len(outcomes):,} rows, {outcomes.prompt_id.nunique():,} prompts, "
        f"{outcomes.model.nunique()} models",
        "rows per source: " + outcomes.source.value_counts().to_string().replace("\n", ", "),
        "prompts per split: "
        + outcomes.drop_duplicates("prompt_id")
        .split.value_counts()
        .to_string()
        .replace("\n", ", "),
        "",
        "mean correctness per model:",
        outcomes.groupby("model").correct.agg(["mean", "count"]).sort_values("mean").to_string(),
        "",
        f"prompts table: {len(prompts):,} rows; difficulty: "
        + prompts.difficulty.value_counts().to_string().replace("\n", ", "),
    ]
    shared = outcomes.groupby("prompt_id").source.nunique()
    lines.append(f"prompts found in more than one source: {(shared > 1).sum():,}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip", nargs="*", default=[], choices=["sprout", "routerbench"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    parts = []
    if "sprout" not in args.skip:
        print("loading SPROUT...")
        parts.append(sprout.to_outcomes(sprout.fetch_raw()))
    if "routerbench" not in args.skip:
        print("loading RouterBench...")
        parts.append(routerbench.to_outcomes(routerbench.fetch_raw("0shot")))
    outcomes = pd.concat(parts, ignore_index=True)

    print("loading RouterArena...")
    prompts = routerarena.to_prompts(routerarena.fetch_raw("full"))

    outcomes.to_parquet(args.out / "outcomes.parquet", index=False)
    prompts.to_parquet(args.out / "prompts.parquet", index=False)
    print(summarise(outcomes, prompts))


if __name__ == "__main__":
    main()
