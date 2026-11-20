"""Run the gateway.

    export ANTHROPIC_API_KEY=... GEMINI_API_KEY=...   # only the providers you use
    python scripts/serve.py --router artifacts/router.pt --port 8080

Then point any OpenAI client at http://localhost:8080/v1 with model="ecoroute/auto", or
Claude Code at the gateway with ANTHROPIC_BASE_URL=http://localhost:8080 (see
docs/claude-code.md).
Only models with a confirmed api_id and an API key present are routed to. With --dry-run
no provider is called: every enabled catalog model can be chosen, and each answer names
the model that would have been used (no API keys, no cost).

--caller-keys is for a gateway hosted for other people: every caller sends their own
provider key (Claude Code does this by itself), the server's keys are never used, and all
models with a confirmed api_id can be routed to.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
import yaml

from ecoroute.gateway import (
    AnthropicBackend,
    EchoBackend,
    EchoMessagesBackend,
    OpenAICompatibleBackend,
    create_app,
)
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
    parser.add_argument(
        "--caller-keys",
        action="store_true",
        help="hosted mode: callers bring their own provider keys; server keys are never used",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="/v1/messages: refuse requests holding secrets instead of redacting them",
    )
    parser.add_argument(
        "--pii-on-tools",
        action="store_true",
        help="/v1/messages: run the PII model over tool results too (slow on CPU, noisy on code)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    providers = yaml.safe_load(args.providers.read_text())["providers"]
    eco = load_gateway_router(
        args.router, args.catalog, providers, policy=args.policy,
        pii_model=not args.no_pii_model, pii_device=args.pii_device, pii_fp16=args.pii_fp16,
        dry_run=args.dry_run, parallel_privacy=args.parallel_privacy,
        caller_keys=args.caller_keys,
    )  # fmt: skip
    print("routing to:", ", ".join(m["name"] for m in eco.catalog))
    if args.dry_run:
        backend, messages = EchoBackend(), EchoMessagesBackend()
    else:
        backend = OpenAICompatibleBackend(providers)
        messages = AnthropicBackend(providers, caller_keys_only=args.caller_keys)
    app = create_app(
        eco,
        backend,
        messages_backend=messages,
        providers=providers,
        redact=not args.no_redact,
        pii_on_tools=args.pii_on_tools,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
