#!/usr/bin/env bash
# EcoRoute installer.
#
# Use a hosted EcoRoute API from Claude Code (nothing heavy is installed):
#   curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh \
#     | bash -s -- --url https://YOUR-ECOROUTE-HOST --key YOUR_ACCESS_KEY
#
# Run your own gateway on this machine and point Claude Code at it:
#   curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh | bash
#
# Options: --url URL, --key KEY (hosted API), --dir DIR (default ~/.ecoroute),
#          --docker (run the gateway in Docker instead of a Python venv),
#          --no-claude-code (skip the Claude Code settings), --remove.
# Running it again updates everything (git pull).
set -euo pipefail

REPO="${ECOROUTE_REPO:-https://github.com/waliddoesthis/ecoroute.git}"
ROUTER_URL="${ECOROUTE_ROUTER_URL:-https://github.com/waliddoesthis/ecoroute/releases/latest/download/router.pt}"
DIR="$HOME/.ecoroute" URL="" KEY="" DOCKER=0 CLAUDE=1 REMOVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --docker) DOCKER=1; shift ;;
    --no-claude-code) CLAUDE=0; shift ;;
    --remove) REMOVE=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "EcoRoute needs $1" >&2; exit 1; }; }
need git
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "EcoRoute needs Python 3.10 or newer" >&2; exit 1; }

if [ "$REMOVE" = 1 ]; then
  [ -f "$DIR/scripts/setup_claude_code.py" ] && "$PY" "$DIR/scripts/setup_claude_code.py" --remove
  say "Claude Code settings removed. Delete $DIR to remove the rest."
  exit 0
fi

if [ -d "$DIR/.git" ]; then
  say "Updating $DIR"; git -C "$DIR" pull --ff-only --quiet
else
  say "Downloading EcoRoute to $DIR"; git clone --quiet --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"

if [ -n "$URL" ]; then
  # Hosted API: the hook and the settings use only the standard library.
  say "Using the hosted EcoRoute API at $URL"
  if ! "$PY" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1].rstrip('/') + '/health', timeout=10)" "$URL" 2>/dev/null; then
    echo "warning: $URL/health did not answer; check the address" >&2
  fi
  [ "$CLAUDE" = 1 ] && "$PY" scripts/setup_claude_code.py --url "$URL" ${KEY:+--key "$KEY"}
  say "Done. Restart Claude Code; each prompt now shows the model EcoRoute picked."
  exit 0
fi

if [ "$DOCKER" = 1 ]; then
  need docker
  say "Building and starting the gateway in Docker (first build takes a few minutes)"
  ECOROUTE_ROUTER_URL="$ROUTER_URL" docker compose up -d --build
else
  say "Installing the gateway in $DIR/.venv (CPU build of PyTorch)"
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  .venv/bin/pip install --quiet -e .
  mkdir -p artifacts
  if [ ! -f artifacts/router.pt ]; then
    say "Downloading the trained router"
    .venv/bin/python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'artifacts/router.pt')" "$ROUTER_URL"
  fi
  cat > ecoroute-serve <<SH
#!/bin/sh
# Start the EcoRoute gateway on http://127.0.0.1:8080 (provider keys come from the env).
cd "$DIR" && exec .venv/bin/python scripts/serve.py --router artifacts/router.pt "\$@"
SH
  chmod +x ecoroute-serve
fi

[ "$CLAUDE" = 1 ] && "$PY" scripts/setup_claude_code.py --url "http://127.0.0.1:8080"
if [ "$DOCKER" = 1 ]; then
  say "Done. The gateway runs in Docker on :8080 and restarts with the machine."
else
  say "Done. Start the gateway with: $DIR/ecoroute-serve   (try --dry-run first)"
fi
say "Then restart Claude Code. Undo with: bash $DIR/install.sh --remove"
