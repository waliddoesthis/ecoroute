"""Time a full routing decision per request, as the gateway makes it.

    python scripts/bench_route.py --router artifacts/router_graph_v6.pt --device 0 --fp16

Reports p50/p95 for the privacy check alone, the embedding plus prediction alone, and the
whole route() call, which runs the first two side by side.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecoroute.confidentiality import CallerPolicyLayer, Detector, NERLayer, RulesLayer
from ecoroute.confidentiality.ner import hf_tagger
from ecoroute.service import EcoRoute


def timed(fn, items) -> np.ndarray:
    out = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        out.append(1000 * (time.perf_counter() - t0))
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--device", type=int, default=None, help="GPU index for the PII model")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    eco = EcoRoute.load(args.router, catalog=args.catalog)
    tagger = hf_tagger(device=args.device, fp16=args.fp16)
    eco.router.detector = Detector([CallerPolicyLayer(), RulesLayer(), NERLayer(tagger)])
    prompts = (
        pd.read_parquet(args.data / "outcomes.parquet", columns=["prompt_id", "prompt"])
        .drop_duplicates("prompt_id")
        .sample(args.n, random_state=1)
        .prompt.tolist()
    )
    eco.route("warm up both models")

    rows = {
        "privacy check": timed(eco.router.detector.classify, prompts),
        "embed + predict": timed(eco.predict, prompts),
        "route (both side by side)": timed(eco.route, prompts),
    }
    for name, ms in rows.items():
        p50, p95 = np.percentile(ms, [50, 95])
        print(f"{name:28s} p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")
    stages = pd.DataFrame([eco.route(p).timings_ms for p in prompts])
    print("\nroute() stages, ms (privacy runs in parallel with embed and predict):")
    print(stages.describe(percentiles=[0.5, 0.95]).loc[["50%", "95%"]].round(1).to_string())


if __name__ == "__main__":
    main()
