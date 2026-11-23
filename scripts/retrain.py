"""Retrain the router on your own graded prompts, so it fits the work you care about.

    python scripts/retrain.py --data my_outcomes.jsonl --out artifacts/router_mine.pt
    python scripts/retrain.py --log-dir logs/            # gateway log + /feedback
    python scripts/serve.py --router artifacts/router_mine.pt

--data is JSONL or CSV with one graded answer per line:

    {"prompt": "Refactor this function ...", "model": "claude-haiku-4-5", "success": true}

`model` is a name from configs/models.yaml and `success` says whether that model's answer
was good enough (true/false, or a score from 0 to 1). Grading several models on the same
prompt teaches the most. --log-dir reads a gateway started with --log-dir, joined with the
feedback sent to POST /feedback.

It holds out 20% of the prompts, fits on the rest, and prints whether the predictions got
closer to what happened on the held-out part (Brier score, lower is better). It then fits
on everything and writes the new router. Runs on a CPU in seconds to minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecoroute.adapt import adapt, read_records
from ecoroute.gateway.logs import read_logged_outcomes
from ecoroute.service import EcoRoute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data", type=Path, nargs="*", default=[], help="JSONL or CSV files")
    parser.add_argument("--log-dir", type=Path, help="a gateway --log-dir with feedback")
    parser.add_argument("--out", type=Path, default=Path("artifacts/router_mine.pt"))
    parser.add_argument("--report", type=Path, help="also write the report as JSON")
    parser.add_argument("--min-per-model", type=int, default=30, help="graded answers per shift")
    parser.add_argument("--k", type=int, default=16, help="similar graded prompts used")
    parser.add_argument("--strength", type=float, default=4.0, help="weight of the prior")
    args = parser.parse_args()

    records = [r for path in args.data for r in read_records(path)]
    if args.log_dir:
        records += read_logged_outcomes(args.log_dir)
    if not records:
        raise SystemExit("no graded answers: pass --data and/or --log-dir")
    eco = EcoRoute.load(args.router, catalog=args.catalog)
    report = adapt(eco, records, min_per_model=args.min_per_model, k=args.k, strength=args.strength)
    print(f"{report['records']} graded answers on {report['prompts']} prompts")
    for m, n in report["per_model"].items():
        shift = report["shifts"].get(m)
        note = f"shift {shift:+.2f}" if shift is not None else f"no shift (< {args.min_per_model})"
        print(f"  {m:28s} {n:6d}  {note}")
    if report["unknown_models_dropped"]:
        print("dropped (not in the catalog):", ", ".join(report["unknown_models_dropped"]))
    t = report["test"]
    if t is None:
        print("Too few prompts to hold some out; fitted on everything without a check.")
    else:
        print(
            f"Held-out check on {t['records']} answers (Brier, lower is better): "
            f"before {t['brier_prior']:.3f}, shifts only {t['brier_shifts']:.3f}, "
            f"adapted {t['brier_adapted']:.3f}"
        )
        if not t["helps"]:
            print(
                "The adapted router did not predict better on held-out prompts; keep the old one."
            )
    eco.save(args.out)
    print(f"saved {args.out}; serve it with: python scripts/serve.py --router {args.out}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
