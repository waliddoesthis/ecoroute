#!/usr/bin/env bash
# Run the shared EcoRoute API on a fresh Ubuntu VM (made for Oracle Cloud Always Free,
# Ampere A1, but any Ubuntu 22.04/24.04 server works, ARM or x86).
#
#   curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/deploy/vm/setup.sh | sudo bash
#
# It installs Docker, starts the gateway in hosted mode (every caller pays with their own
# provider key; only holders of an access key get in), and puts Caddy in front for HTTPS
# with a free certificate. Without a domain it uses <ip>.sslip.io, which points at this
# server. Re-run it to update. Options (environment variables):
#   ECOROUTE_DOMAIN=api.example.com   your own domain (its DNS must point here)
#   ECOROUTE_ACCESS_KEYS=k1,k2        access keys to use (default: one is generated)
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
DIR=/opt/ecoroute

say "Installing Docker and git"
if ! command -v docker >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl git >/dev/null
  curl -fsSL https://get.docker.com | sh >/dev/null
fi
command -v git >/dev/null || apt-get install -y -qq git >/dev/null

# Oracle's Ubuntu images block every port but SSH in iptables (on top of the cloud
# firewall, the security list, which you open in the console).
if command -v iptables >/dev/null && iptables -L INPUT -n 2>/dev/null | grep -q REJECT; then
  say "Opening ports 80 and 443 in the VM's firewall"
  for p in 80 443; do
    iptables -C INPUT -p tcp --dport "$p" -j ACCEPT 2>/dev/null \
      || iptables -I INPUT 5 -p tcp --dport "$p" -j ACCEPT
  done
  if command -v netfilter-persistent >/dev/null; then netfilter-persistent save >/dev/null; fi
fi

if [ -d "$DIR/.git" ]; then say "Updating $DIR"; git -C "$DIR" pull -q --ff-only
else say "Downloading EcoRoute"; git clone -q --depth 1 https://github.com/waliddoesthis/ecoroute.git "$DIR"; fi
cd "$DIR"

ENV_FILE="$DIR/deploy/vm/.env"
if [ ! -f "$ENV_FILE" ]; then
  KEYS="${ECOROUTE_ACCESS_KEYS:-eco-$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)}"
  IP="$(curl -fsS https://api.ipify.org || hostname -I | awk '{print $1}')"
  DOMAIN="${ECOROUTE_DOMAIN:-$(echo "$IP" | tr . -).sslip.io}"
  umask 077
  printf 'ECOROUTE_ACCESS_KEYS=%s\nECOROUTE_DOMAIN=%s\n' "$KEYS" "$DOMAIN" > "$ENV_FILE"
fi
# shellcheck disable=SC1090
. "$ENV_FILE"

say "Building and starting the gateway (the first build takes 10-20 minutes)"
docker compose -f deploy/vm/compose.yml --env-file "$ENV_FILE" up -d --build

say "Waiting for the gateway"
for _ in $(seq 1 90); do
  curl -fsS "http://127.0.0.1:8080/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS "http://127.0.0.1:8080/health" >/dev/null || { echo "gateway did not start: docker compose -f $DIR/deploy/vm/compose.yml logs" >&2; exit 1; }

cat <<MSG

EcoRoute API is running at  https://$ECOROUTE_DOMAIN
(HTTPS starts working once ports 80 and 443 are open in Oracle's security list.)

Access keys (give one to each person; stored in $ENV_FILE):
  $ECOROUTE_ACCESS_KEYS

Each person connects Claude Code with:
  curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh \\
    | bash -s -- --url https://$ECOROUTE_DOMAIN --key <their key>

Logs: docker compose -f $DIR/deploy/vm/compose.yml logs -f
MSG
