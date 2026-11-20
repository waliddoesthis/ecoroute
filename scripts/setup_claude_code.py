"""Point Claude Code at an EcoRoute gateway, or undo it.

    python scripts/setup_claude_code.py                         # local gateway on :8080
    python scripts/setup_claude_code.py --url https://ecoroute.example.com --key KEY
    python scripts/setup_claude_code.py --project               # this repo only
    python scripts/setup_claude_code.py --remove

Writes two things into Claude Code's settings.json (~/.claude/settings.json, or
.claude/settings.json with --project), keeping everything else and a .bak copy:
  env.ANTHROPIC_BASE_URL       every Claude Code request goes through the gateway
  hooks.UserPromptSubmit       ecoroute_hook.py shows the routing decision for each prompt
With --key, the gateway access key is sent as X-EcoRoute-Key (ANTHROPIC_CUSTOM_HEADERS).
Your Anthropic key or Claude login is used as before and passed through to Anthropic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "integrations" / "claude-code" / "ecoroute_hook.py"
MARK = "ecoroute_hook.py"


def is_ours(entry: dict) -> bool:
    return any(MARK in h.get("command", "") for h in entry.get("hooks", []))


def apply(settings: dict, url: str | None, key: str | None, remove: bool) -> dict:
    env = settings.setdefault("env", {})
    hooks = settings.setdefault("hooks", {})
    entries = [e for e in hooks.get("UserPromptSubmit", []) if not is_ours(e)]
    for name in ("ANTHROPIC_BASE_URL", "ECOROUTE_URL", "ECOROUTE_KEY"):
        env.pop(name, None)
    headers = [
        h for h in env.get("ANTHROPIC_CUSTOM_HEADERS", "").split("\n")
        if h and not h.lower().startswith("x-ecoroute-key")
    ]  # fmt: skip
    if not remove:
        env["ANTHROPIC_BASE_URL"] = url
        env["ECOROUTE_URL"] = url  # read by the hook
        if key:
            env["ECOROUTE_KEY"] = key
            headers.append(f"X-EcoRoute-Key: {key}")
        cmd = f'"{sys.executable}" "{HOOK}"'
        entries.append({"hooks": [{"type": "command", "command": cmd, "timeout": 10}]})
    if headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(headers)
    else:
        env.pop("ANTHROPIC_CUSTOM_HEADERS", None)
    if entries:
        hooks["UserPromptSubmit"] = entries
    else:
        hooks.pop("UserPromptSubmit", None)
    for k in ("env", "hooks"):
        if not settings[k]:
            del settings[k]
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="gateway address")
    parser.add_argument("--key", help="gateway access key, for a hosted gateway")
    parser.add_argument("--project", action="store_true", help="write .claude/settings.json here")
    parser.add_argument("--settings", type=Path, help="settings file to edit (overrides both)")
    parser.add_argument("--remove", action="store_true", help="take EcoRoute out again")
    args = parser.parse_args()
    path = args.settings or (
        Path(".claude/settings.json") if args.project else Path.home() / ".claude/settings.json"
    )
    settings = json.loads(path.read_text()) if path.exists() else {}
    if path.exists():
        shutil.copy(path, path.with_suffix(".json.bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(apply(settings, args.url, args.key, args.remove), indent=2) + "\n")
    if args.remove:
        print(f"EcoRoute removed from {path}")
    else:
        print(f"Claude Code now routes through {args.url} ({path}). Restart Claude Code.")


if __name__ == "__main__":
    main()
