"""Route a few example prompts with a trained router and print each decision.

python scripts/route_demo.py --router artifacts/router.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecoroute.service import EcoRoute

EXAMPLES = [
    ("Translate 'thank you very much' into French.", None),
    ("What is the capital of Australia?", None),
    ("Write a Python function that merges overlapping intervals, with tests.", None),
    ("Prove that there are infinitely many primes of the form 4k+3.", None),
    ("A train leaves at 3pm at 80 km/h; another at 4pm at 100 km/h. When does it catch up?", None),
    ("Summarise our Q3 board deck in five bullet points.", {"sensitivity_label": "Confidential"}),
    ("Email jane.doe@example.com a reminder about Friday's meeting.", None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--policy", default="balanced")
    args = parser.parse_args()
    eco = EcoRoute.load(args.router, catalog=args.catalog, policy=args.policy)
    for prompt, context in EXAMPLES:
        d = eco.route(prompt, context=context)
        probs = ", ".join(f"{c.name}={c.p_success:.2f}" for c in d.candidates)
        print(f"\n> {prompt}\n  -> {d.model}  [{d.level}, difficulty {d.difficulty:.2f}]")
        print("  P(correct): " + probs)
        print("  " + d.explain().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
