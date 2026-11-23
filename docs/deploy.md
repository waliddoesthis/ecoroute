# Deploying EcoRoute

There are three ways to use EcoRoute. Pick the first one that fits.

| You want | Run | What it installs |
|---|---|---|
| To use an EcoRoute API someone hosts | `install.sh --url URL --key KEY` | Claude Code settings and a small hook (Python standard library only) |
| Your own gateway on your machine | `install.sh` (or `install.sh --docker`) | The gateway in a venv (CPU PyTorch) or in Docker, plus the Claude Code settings |
| To host the API for other people | `docker compose up -d` with hosted mode on | The gateway in Docker, behind HTTPS |

```bash
# Use a hosted API
curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh \
  | bash -s -- --url https://YOUR-ECOROUTE-HOST --key YOUR_ACCESS_KEY

# Run your own
curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh | bash
```

Running `install.sh` again updates everything with `git pull`. `install.sh --remove` takes
EcoRoute out of Claude Code.

## The router file

The trained router (`artifacts/router.pt`, about 84 MB) is not in git. The installer and
the Docker image download it from the latest GitHub release
(`releases/latest/download/router.pt`). Set `ECOROUTE_ROUTER_URL` to use another file, for
example one you retrained on your own data (`scripts/retrain.py`).

## Hosting the API for others

Run the gateway in hosted mode:

```bash
export ECOROUTE_ACCESS_KEYS="key-for-alice,key-for-bob"   # who may use it
ECOROUTE_ARGS=--caller-keys docker compose up -d
```

Hosted mode changes three things:

- **Everyone pays with their own provider key, and the server's keys are never used.**
  Claude Code sends its user's Anthropic key or login as usual, and the gateway passes it
  on. OpenAI-format apps send one header per provider they use, such as
  `X-Provider-Key-OpenAI: sk-...` or `X-Provider-Key-Google: ...`. The router only picks
  among providers the caller sent a key for. Keys are used for that one request and never
  stored.
- **Only callers with an access key get in.** They send it as `X-EcoRoute-Key`. Claude
  Code does this through `ANTHROPIC_CUSTOM_HEADERS`, which `install.sh --key` sets up.
  OpenAI-format apps can also send it as their API key (the `Authorization` bearer).
- **No logging by default.** `--log-dir` writes prompt text to disk. Only turn it on with
  your users' agreement.

Put the gateway behind HTTPS, because keys travel in the headers. Any reverse proxy works,
and so does a platform that gives you HTTPS by itself.

Clients then point at it:

```python
from openai import OpenAI
client = OpenAI(base_url="https://YOUR-ECOROUTE-HOST/v1", api_key="YOUR_ACCESS_KEY",
                default_headers={"X-Provider-Key-OpenAI": "sk-...",
                                 "X-Provider-Key-Google": "..."})
client.chat.completions.create(model="ecoroute/auto", messages=[...])
```

```bash
# Claude Code, or any Anthropic SDK app
export ANTHROPIC_BASE_URL=https://YOUR-ECOROUTE-HOST
export ANTHROPIC_CUSTOM_HEADERS="X-EcoRoute-Key: YOUR_ACCESS_KEY"
```

### Where to host it

The image needs about 2 GB of memory and no GPU. On a 4-core CPU, a routing decision for
a short prompt took about 0.2 s, most of it the PII model (against 74 ms on a T4 GPU). A
first scan of a large file takes longer, and repeated blocks come from the cache. Next to
an LLM's answer time, that is usually small.

- **Hugging Face Spaces (free CPU), the default.** Upload the two files in
  [`deploy/hf-space/`](../deploy/hf-space/) to a new Docker Space and add
  `ECOROUTE_ACCESS_KEYS` as a secret. That folder's README has the steps. Free Spaces
  sleep when idle, and the first request after a sleep is slow.
- **A small VM.** Install Docker, run `docker compose up -d`, and put Caddy or nginx in
  front for HTTPS. `restart: unless-stopped` brings the gateway back after a reboot.
- **Your own computer, through a tunnel.** This works for a few users, but the gateway is
  only reachable while the computer is on.

Whatever you choose, every prompt your users send passes through that machine. Choose a
host you would trust with that data.

## What is real and what is not yet

- Inside Claude Code, the gateway picks among Claude models only (see
  [claude-code.md](claude-code.md)).
- The ChatGPT and Gemini apps and websites cannot be pointed at a gateway. EcoRoute works
  with their APIs, and with any tool that lets you set an OpenAI- or Anthropic-compatible
  base URL.
- The gateway has access keys but no per-user rate limits or billing. Put it behind an
  API gateway if you need those.
