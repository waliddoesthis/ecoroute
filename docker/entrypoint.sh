#!/bin/sh
# Start the gateway. Extra arguments go to scripts/serve.py (e.g. --caller-keys, --dry-run).
set -e
ROUTER="${ECOROUTE_ROUTER:-artifacts/router.pt}"
if [ ! -f "$ROUTER" ]; then
  if [ -z "$ECOROUTE_ROUTER_URL" ]; then
    echo "No router at $ROUTER: mount one or set ECOROUTE_ROUTER_URL" >&2
    exit 1
  fi
  echo "Downloading the router from $ECOROUTE_ROUTER_URL"
  python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
    "$ECOROUTE_ROUTER_URL" "$ROUTER"
fi
exec python scripts/serve.py --router "$ROUTER" $ECOROUTE_ARGS "$@"
