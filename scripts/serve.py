"""Run the gateway.

    export ANTHROPIC_API_KEY=... GEMINI_API_KEY=...   # only the providers you use
    python scripts/serve.py --router artifacts/router.pt --port 8080

Then point any OpenAI client at http://localhost:8080/v1 with model="ecoroute/auto".
Only models with a confirmed api_id and an API key present are routed to. With --dry-run
no provider is called: every enabled catalog model can be chosen, and each answer names
the model that would have been used (no API keys, no cost).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
import yaml

from ecoroute.gateway import EchoBackend, OpenAICompatibleBackend, create_app
from ecoroute.gateway.load import load_gateway_router


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
        "--dry-run", action="store_true", help="call no provider; answers name the chosen model"
    )
    parser.add_argument(
        "--parallel-privacy",
        action="store_true",
        help="run the privacy check beside the embedding (helps when they use different "
        "devices; on one shared GPU it is slower)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    providers = yaml.safe_load(args.providers.read_text())["providers"]
    eco = load_gateway_router(
        args.router, args.catalog, providers, policy=args.policy,
        pii_model=not args.no_pii_model, pii_device=args.pii_device, pii_fp16=args.pii_fp16,
        dry_run=args.dry_run, parallel_privacy=args.parallel_privacy,
    )  # fmt: skip
    print("routing to:", ", ".join(m["name"] for m in eco.catalog))
    backend = EchoBackend() if args.dry_run else OpenAICompatibleBackend(providers)
    uvicorn.run(create_app(eco, backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
