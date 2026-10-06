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
    parser.add_argument("--quantize-pii", action="store_true", help="8-bit PII model on CPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    providers = yaml.safe_load(args.providers.read_text())["providers"]
    full = EcoRoute.load(args.router, catalog=args.catalog)
    live = deployable(full.catalog, providers)
    if not live:
        raise SystemExit("no deployable models: set api_id in models.yaml and the API key env vars")
    layers = [CallerPolicyLayer(), RulesLayer()]
    if not args.no_pii_model:
        layers.append(NERLayer(hf_tagger(quantize=args.quantize_pii)))
    router = Router(live, detector=Detector(layers), policy=args.policy)
    eco = EcoRoute(full.predictor.base, full.encoder_name, live, router=router)
    print("routing to:", ", ".join(m["name"] for m in live))
    uvicorn.run(create_app(eco, OpenAICompatibleBackend(providers)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
