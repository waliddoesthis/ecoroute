---
title: EcoRoute API
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8080
pinned: false
license: mit
---

# EcoRoute API

A shared EcoRoute gateway. It routes each request to the cheapest model predicted to
handle it, after a privacy check. Callers pay with their own provider keys, and the
Space never stores them. Source: https://github.com/waliddoesthis/ecoroute

## Set it up (once)

1. On Hugging Face, create a **New Space**, choose the **Docker** SDK (blank template)
   and the free **CPU basic** hardware. Creating a Docker Space needs a paid plan (PRO for
   a personal account), even though the hardware is free.
2. Upload this folder's `Dockerfile` and `README.md` to the Space.
3. Under **Settings → Variables and secrets**, add a secret `ECOROUTE_ACCESS_KEYS` with
   one or more access keys, comma separated (one per person you let in).
4. Wait for the build (about 10 minutes). `https://<user>-<space>.hf.space/health`
   answers `{"status":"ok"}`.

To update after a new release, use **Settings → Factory rebuild**.

## Use it

```bash
# Claude Code
curl -fsSL https://raw.githubusercontent.com/waliddoesthis/ecoroute/main/install.sh \
  | bash -s -- --url https://<user>-<space>.hf.space --key <access key>
```

OpenAI-format apps use `base_url="https://<user>-<space>.hf.space/v1"`,
`api_key="<access key>"` and `model="ecoroute/auto"`, with one
`X-Provider-Key-<provider>` header per provider key (see docs/deploy.md).

Free Spaces sleep after a while with no requests. The first request after that takes
about a minute while the Space starts.
