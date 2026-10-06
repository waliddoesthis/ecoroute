"""Scale measured per-request savings to company volumes (the tables in docs/impact.md).

    python scripts/impact.py

Every input is an assumption stated below and printed with the result. The saving rates
come from held-out test results (run 36 for cost against GPT-4o, run 38 for energy and
cost on the deployed catalog); everything else is a scenario, not a measurement.
"""

from __future__ import annotations

import argparse

# GPT-4o list price, USD per 1M tokens (the reference model of the measured savings).
GPT4O_PRICE = (2.50, 10.00)
# A typical enterprise request: about a page of context in, a few paragraphs out.
TOKENS = (1000, 400)
# Measured on SPROUT test prompts against always using GPT-4o (run 36).
COST_SAVED = {"quality": 0.342, "balanced": 0.697, "eco": 0.750}
# Grid carbon intensity, kg CO2 per kWh (a world-average order of magnitude).
KG_CO2_PER_KWH = 0.4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=float, nargs="+", default=[1e6, 10e6, 100e6])
    parser.add_argument("--tokens-in", type=int, default=TOKENS[0])
    parser.add_argument("--tokens-out", type=int, default=TOKENS[1])
    parser.add_argument(
        "--wh", type=float, nargs=2, metavar=("REFERENCE", "ROUTED"), default=None,
        help="Wh per request always using the top model and with EcoRoute (run 38)",
    )  # fmt: skip
    args = parser.parse_args()

    per_request = (args.tokens_in * GPT4O_PRICE[0] + args.tokens_out * GPT4O_PRICE[1]) / 1e6
    print(f"assumed request: {args.tokens_in} tokens in, {args.tokens_out} out")
    print(f"always GPT-4o: ${per_request:.5f} per request (list price {GPT4O_PRICE} per 1M)\n")
    print("| Requests / month | Always GPT-4o / year | " + " | ".join(
        f"{p} saves / year" for p in COST_SAVED) + " |")  # fmt: skip
    print("|---|---|" + "---|" * len(COST_SAVED))
    for v in args.volumes:
        year = per_request * v * 12
        cells = " | ".join(f"${year * s:,.0f}" for s in COST_SAVED.values())
        print(f"| {v / 1e6:,.0f}M | ${year:,.0f} | {cells} |")

    if args.wh:
        ref, routed = args.wh
        print(f"\nenergy: {ref:.3f} Wh per request always top model, {routed:.3f} with EcoRoute "
              f"({100 * (1 - routed / ref):.0f}% less); {KG_CO2_PER_KWH} kg CO2/kWh")  # fmt: skip
        print("| Requests / month | Energy saved / year | CO2 avoided / year |")
        print("|---|---|---|")
        for v in args.volumes:
            kwh = (ref - routed) * v * 12 / 1000
            print(f"| {v / 1e6:,.0f}M | {kwh:,.0f} kWh | {kwh * KG_CO2_PER_KWH / 1000:,.1f} t |")


if __name__ == "__main__":
    main()
