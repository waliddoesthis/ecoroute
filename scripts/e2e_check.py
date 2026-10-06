"""End-to-end check of the gateway on a trained router: privacy filter, graph routing,
explanations and the OpenAI-compatible API, with no provider called and no cost.

    python scripts/e2e_check.py --router artifacts/router_graph_v14.pt

It loads the router as serve.py does (full privacy check, PII model included unless
--no-pii-model), answers with EchoBackend, sends real requests through the HTTP app and
prints one line per check. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ecoroute.confidentiality import Level
from ecoroute.gateway import EchoBackend, create_app
from ecoroute.gateway.load import load_gateway_router

AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"  # AWS's documented example key, split for scanners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.pt"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--providers", type=Path, default=Path("configs/providers.yaml"))
    parser.add_argument("--no-pii-model", action="store_true")
    parser.add_argument("--pii-device", type=int, default=None)
    args = parser.parse_args()

    providers = yaml.safe_load(args.providers.read_text())["providers"]
    eco = load_gateway_router(
        args.router, args.catalog, providers, pii_model=not args.no_pii_model,
        pii_device=args.pii_device, dry_run=True,
    )  # fmt: skip
    client = TestClient(create_app(eco, EchoBackend()))
    clearance = {m["name"]: Level.parse(m.get("clearance", "public")) for m in eco.catalog}
    price = {m["name"]: m.get("price_out_per_mtok") or 0.0 for m in eco.catalog}
    results, times = [], []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))

    def chat(messages, model="ecoroute/auto", headers=None, **kw):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        r = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, **kw},
            headers=headers or {},
        )
        if t := r.headers.get("x-ecoroute-time-ms"):
            times.append(float(dict(p.split("=") for p in t.split(", "))["total"]))
        return r

    def routed(r) -> tuple[str, Level]:
        return r.headers["x-ecoroute-model"], Level.parse(r.headers["x-ecoroute-level"])

    # 1. The API looks like OpenAI's.
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    check("models list has ecoroute/auto and the catalog", ids[0] == "ecoroute/auto"
          and set(ids[1:]) == set(clearance), f"{len(ids) - 1} models")  # fmt: skip

    # 2. An easy, harmless prompt leaves for a cheap external model, with an explanation.
    r = chat("Translate 'thank you very much' into French.")
    model, level = routed(r)
    answer = r.json()["choices"][0]["message"]["content"]
    check("easy prompt is answered", r.status_code == 200 and model in answer, model)
    check("easy prompt goes to a cheaper model than the priciest",
          price[model] < max(price.values()), f"{model}, {level}")  # fmt: skip
    check("decision is explained in headers",
          bool(r.headers.get("x-ecoroute-reason")) and "x-ecoroute-difficulty" in r.headers,
          r.headers.get("x-ecoroute-reason", "")[:90])  # fmt: skip

    # 3. Privacy filter: sensitive prompts stay on models cleared for them.
    cases = [
        ("secret in an earlier message stays local", [
            {"role": "system", "content": f"Use the key {AWS} for S3."},
            {"role": "user", "content": "Why does my upload fail?"},
        ], {}, Level.RESTRICTED),
        ("email address is confidential",
         "Email jane.doe@example.com a reminder about Friday's meeting.", {}, Level.CONFIDENTIAL),
        ("bare account number is confidential",
         "Move 200 euros to 40217710938 tomorrow morning.", {}, Level.CONFIDENTIAL),
        ("caller's Confidential label is honoured", "Summarise our Q3 board deck.",
         {"X-EcoRoute-Sensitivity": "Confidential"}, Level.CONFIDENTIAL),
    ]  # fmt: skip
    if not args.no_pii_model:
        cases.append(
            ("name and home address found by the PII model",
             "Write a letter to Sarah Johnson, 42 Elm Street, Springfield, about her overdue "
             "invoice.", {}, Level.CONFIDENTIAL)
        )  # fmt: skip
    for name, messages, headers, want in cases:
        r = chat(messages, headers=headers)
        model, level = routed(r)
        check(name, r.status_code == 200 and level >= want and clearance[model] >= level,
              f"{level} -> {model}")  # fmt: skip
    leaked = AWS in json.dumps(dict(r.headers)) + r.text
    check("matched secrets are never echoed", not leaked)

    # 4. A named model still goes through the privacy check.
    r = chat(f"debug this: {AWS}", model="claude-opus-5-5")
    check("uncleared named model is refused", r.status_code == 403, str(r.status_code))
    check("unknown model is refused", chat("hi", model="no-such-model").status_code == 404)

    # 5. Profiles: quality pays at least as much as eco on a hard prompt.
    hard = "Prove that there are infinitely many primes of the form 4k+3."
    picks = {}
    for policy in ("eco", "balanced", "quality"):
        d = client.post("/route", json={"prompt": hard}, headers={"X-EcoRoute-Policy": policy})
        body = d.json()
        cost = next(c["cost_usd"] for c in body["candidates"] if c["name"] == body["model"])
        picks[policy] = (body["model"], cost)
    check("quality spends at least as much as eco", picks["quality"][1] >= picks["eco"][1],
          ", ".join(f"{k} {v[0]}" for k, v in picks.items()))  # fmt: skip
    check("unknown profile is a 400", client.post(
        "/route", json={"prompt": "hi"}, headers={"X-EcoRoute-Policy": "fastest"}
    ).status_code == 400)  # fmt: skip

    # 6. /route explains every step without calling a model.
    steps = client.post("/route", json={"prompt": hard}).json()["explanation"]
    text = " ".join(steps)
    check("route explanation covers privacy, graph and quality floor",
          "Confidentiality" in text and "Graph" in text and "Quality floor" in text,
          f"{len(steps)} steps")  # fmt: skip

    # 7. A coding prompt (the case the fallback margin was added for).
    r = chat("Write a Python function that merges overlapping intervals, with tests.")
    model, _ = routed(r)
    print(f"info  coding prompt -> {model}: {r.headers.get('x-ecoroute-reason', '')}")

    # 8. Streaming passes through as server-sent events.
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "ecoroute/auto", "stream": True,
        "messages": [{"role": "user", "content": "Say hello."}],
    }) as s:  # fmt: skip
        raw = "".join(s.iter_text())
    check("streaming ends with [DONE]", s.status_code == 200 and raw.strip().endswith("[DONE]"))

    if times:
        print(f"info  routing time per request: p50 {statistics.median(times):.1f} ms, "
              f"max {max(times):.1f} ms over {len(times)} requests (first includes loading)")  # fmt: skip
    failed = results.count(False)
    print(f"\n{len(results) - failed} of {len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
