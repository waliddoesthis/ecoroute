"""Sending the chosen request to a provider."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

import httpx


class Backend(Protocol):
    """keys (hosted gateways only): provider name -> the caller's own API key."""

    async def complete(self, model: Mapping[str, Any], body: dict, **kw) -> dict: ...

    def stream(self, model: Mapping[str, Any], body: dict, **kw) -> AsyncIterator[bytes]: ...


def deployable(
    catalog: list[Mapping[str, Any]],
    providers: Mapping[str, Mapping[str, Any]],
    env: Mapping[str, str] | None = None,
    require_key: bool = True,
) -> list[Mapping[str, Any]]:
    """Catalog models the gateway can actually call: a known provider, a confirmed
    api_id, and the provider's API key present in the environment (unless require_key is
    False, for a hosted gateway where every caller sends their own key)."""
    env = os.environ if env is None else env
    out = []
    for m in catalog:
        p = providers.get(m.get("provider", ""))
        if not p or not m.get("api_id"):
            continue
        key_env = p.get("api_key_env")
        if require_key and key_env and not env.get(key_env):
            continue
        out.append(m)
    return out


class OpenAICompatibleBackend:
    def __init__(
        self,
        providers: Mapping[str, Mapping[str, Any]],
        env: Mapping[str, str] | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        caller_keys_only: bool = False,
    ) -> None:
        self.providers = providers
        self.env = os.environ if env is None else env
        # Hosted for others: only the caller's keys are used, never the server's.
        self.caller_keys_only = caller_keys_only
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    def _request(
        self, model: Mapping[str, Any], body: dict, keys: Mapping[str, str] | None = None
    ) -> tuple[str, dict, dict]:
        p = self.providers[model["provider"]]
        headers = {"content-type": "application/json"}
        key = (keys or {}).get(model["provider"].lower())
        if key is None and p.get("api_key_env") and not self.caller_keys_only:
            key = self.env.get(p["api_key_env"])
        if key:
            headers["authorization"] = f"Bearer {key}"
        url = p["base_url"].rstrip("/") + "/chat/completions"
        return url, headers, {**body, "model": model["api_id"]}

    async def complete(self, model, body, keys=None):
        url, headers, payload = self._request(model, body, keys)
        r = await self.client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    async def stream(self, model, body, keys=None):
        url, headers, payload = self._request(model, body, keys)
        async with self.client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk


class EchoBackend:
    """Calls no provider: answers with the model that would have been called. For trying
    the gateway, and for end-to-end checks, without API keys or cost."""

    def _text(self, model: Mapping[str, Any]) -> str:
        return f"[dry run] {model['name']} would answer this."

    async def complete(self, model, body, **kw):
        return {
            "id": "ecoroute-dry-run",
            "object": "chat.completion",
            "model": model["name"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._text(model)},
                    "finish_reason": "stop",
                }
            ],
        }

    async def stream(self, model, body, **kw):
        chunk = {
            "id": "ecoroute-dry-run",
            "object": "chat.completion.chunk",
            "model": model["name"],
            "choices": [{"index": 0, "delta": {"content": self._text(model)}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
