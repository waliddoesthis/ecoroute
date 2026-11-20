#!/usr/bin/env python3
"""Claude Code hook (UserPromptSubmit): show which model EcoRoute picks for each prompt.

Claude Code runs this before every prompt it sends. It asks the gateway's /route endpoint
for the decision (no model is called) and prints one line in Claude Code, for example:

    EcoRoute: claude-haiku-4-5 (easy, 75% cheaper than claude-opus-5-5)

The gateway does the actual routing when Claude Code sends the request, so this hook only
informs: it never blocks or changes a prompt, and it stays silent if the gateway is down.
Standard library only. Gateway address: ECOROUTE_URL, else ANTHROPIC_BASE_URL.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def summary(d: dict) -> str:
    diff = d.get("difficulty")
    bits = []
    if diff is not None:
        bits.append("easy" if diff < 0.3 else "hard" if diff > 0.6 else "medium")  # as the gateway
    allowed = [c for c in d.get("candidates", []) if c.get("allowed")]
    chosen = next((c for c in allowed if c["name"] == d["model"]), None)
    priciest = max(allowed, key=lambda c: c.get("cost_usd") or 0, default=None)
    if chosen and priciest and priciest["name"] != chosen["name"] and priciest["cost_usd"]:
        saved = 100 * (1 - chosen["cost_usd"] / priciest["cost_usd"])
        bits.append(f"{saved:.0f}% cheaper than {priciest['name']}")
    return f"EcoRoute: {d['model']}" + (f" ({', '.join(bits)})" if bits else "")


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    prompt = event.get("prompt") or ""
    base = os.environ.get("ECOROUTE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    if not prompt.strip() or not base:
        return 0
    headers = {"content-type": "application/json", "x-ecoroute-api": "messages"}
    if key := os.environ.get("ECOROUTE_KEY"):
        headers["x-ecoroute-key"] = key
    body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/route", data=body, headers=headers)
    try:
        with urllib.request.urlopen(
            req, timeout=float(os.environ.get("ECOROUTE_HOOK_TIMEOUT", 5))
        ) as r:
            message = summary(json.load(r))
    except urllib.error.HTTPError as e:
        if e.code != 403:
            return 0
        message = (
            "EcoRoute: this prompt holds data no cloud model is cleared for; "
            "the gateway redacts it before anything leaves"
        )
    except (OSError, ValueError, KeyError):
        return 0  # gateway unreachable: say nothing, Claude Code carries on
    print(json.dumps({"systemMessage": message}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
