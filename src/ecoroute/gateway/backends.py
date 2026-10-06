"""Sending the chosen request to a provider."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

import httpx


class Backend(Protocol):
    async def complete(self, model: Mapping[str, Any], body: dict) -> dict: ...

    def stream(self, model: Mapping[str, Any], body: dict) -> AsyncIterator[bytes]: ...


def deployable(
    catalog: list[Mapping[str, Any]],
    providers: Mapping[str, Mapping[str, Any]],
    env: Mapping[str, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Catalog models the gateway can actually call: a known provider, a confirmed
    api_id, and the provider's API key present in the environment."""
    env = os.environ if env is None else env
    out = []
    for m in catalog:
        p = providers.get(m.get("provider", ""))
        if not p or not m.get("api_id"):
            continue
        key_env = p.get("api_key_env")
        if key_env and not env.get(key_env):
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
    ) -> None:
        self.providers = providers
        self.env = os.environ if env is None else env
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    def _request(self, model: Mapping[str, Any], body: dict) -> tuple[str, dict, dict]:
        p = self.providers[model["provider"]]
        headers = {"content-type": "application/json"}
        if p.get("api_key_env"):
            headers["authorization"] = f"Bearer {self.env[p['api_key_env']]}"
        url = p["base_url"].rstrip("/") + "/chat/completions"
        return url, headers, {**body, "model": model["api_id"]}

    async def complete(self, model, body):
        url, headers, payload = self._request(model, body)
        r = await self.client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    async def stream(self, model, body):
        url, headers, payload = self._request(model, body)
        async with self.client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk
