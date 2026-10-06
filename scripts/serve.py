"""Run the gateway.

    export ANTHROPIC_API_KEY=... GEMINI_API_KEY=...   # only the providers you use
    python scripts/serve.py --router artifacts/router.pt --port 8080

Then point any OpenAI client at http://localhost:8080/v1 with model="ecoroute/auto".
Only models with a confirmed api_id and an API key present are routed to.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
import yaml

from ecoroute.confidentiality import CallerPolicyLayer, Detector, NERLayer, RulesLayer
from ecoroute.confidentiality.ner import hf_tagger
from ecoroute.gateway import OpenAICompatibleBackend, create_app, deployable
from ecoroute.routing import Router
from ecoroute.service import EcoRoute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--providers", type=Path, default=Path("configs/providers.yaml"))
    parser.add_argument("--policy", default="balanced")
    parser.add_argument("--no-pii-model", action="store_true", help="rules only, no PII model")
    parser.add_argument("--pii-device", type=int, default=None, help="GPU index for the PII model")
    parser.add_argument("--pii-fp16", action="store_true", help="half-precision PII model (GPU)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--parallel-privacy",
        action="store_true",
        help="run the privacy check beside the embedding (helps when they use different "
        "devices; on one shared GPU it is slower)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    providers = yaml.safe_load(args.providers.read_text())["providers"]
    full = EcoRoute.load(args.router, catalog=args.catalog)
    live = deployable(full.catalog, providers)
    if not live:
        raise SystemExit("no deployable models: set api_id in models.yaml and the API key env vars")
    layers = [CallerPolicyLayer(), RulesLayer()]
    if not args.no_pii_model:
        layers.append(NERLayer(hf_tagger(device=args.pii_device, fp16=args.pii_fp16)))
    # Keep the quality floors the router was trained with.
    router = Router(
        live, detector=Detector(layers), policy=args.policy, profiles=full.router.profiles
    )
    eco = EcoRoute(
        full.predictor.base, full.encoder_name, live, router=router, taus=full.taus, margins=full.margins,
        parallel_privacy=args.parallel_privacy,
    )  # fmt: skip
    print("routing to:", ", ".join(m["name"] for m in live))
    uvicorn.run(create_app(eco, OpenAICompatibleBackend(providers)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
