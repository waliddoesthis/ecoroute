"""A gateway hosted for others: access keys and each caller's own provider keys."""

import httpx
import numpy as np
from fastapi.testclient import TestClient

from ecoroute.gateway import OpenAICompatibleBackend, create_app
from ecoroute.predictors import ModelMeanPredictor
from ecoroute.service import EcoRoute

PROVIDERS = {
    "openai": {"base_url": "https://openai.test/v1", "api_key_env": "OPENAI_API_KEY"},
    "google": {"base_url": "https://google.test/v1", "api_key_env": "GEMINI_API_KEY"},
}
CATALOG = [
    {"name": "flash", "provider": "google", "api_id": "g", "clearance": "internal",
     "price_out_per_mtok": 1.0, "anchor": {"model": "a"}},
    {"name": "gpt", "provider": "openai", "api_id": "o", "clearance": "internal",
     "price_out_per_mtok": 10.0, "anchor": {"model": "b"}},
]  # fmt: skip


def make(calls, **kw):
    def upstream(request):
        calls.append((request.url.host, request.headers.get("authorization")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    base = ModelMeanPredictor().fit(np.zeros((2, 2)), np.ones((2, 2)), ["a", "b"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = lambda texts: np.zeros((len(texts), 2), np.float32)
    backend = OpenAICompatibleBackend(
        PROVIDERS, env={"GEMINI_API_KEY": "server-g"}, caller_keys_only=True,
        transport=httpx.MockTransport(upstream),
    )  # fmt: skip
    return TestClient(create_app(eco, backend, providers=PROVIDERS, caller_keys=True, **kw))


def chat(client, headers):
    body = {"model": "ecoroute/auto", "messages": [{"role": "user", "content": "hi"}]}
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_only_providers_the_caller_has_keys_for_and_their_key_is_used():
    calls = []
    r = chat(make(calls), {"x-provider-key-openai": "sk-caller"})
    assert r.status_code == 200 and r.headers["x-ecoroute-model"] == "gpt"
    assert calls == [("openai.test", "Bearer sk-caller")]  # never the server's google key


def test_access_keys():
    calls = []
    client = make(calls, access_keys=["team-key"])
    assert client.get("/health").status_code == 200
    assert chat(client, {"x-provider-key-google": "g1"}).status_code == 401
    ok = {"x-provider-key-google": "g1", "x-ecoroute-key": "team-key"}
    assert chat(client, ok).status_code == 200
    bearer = {"x-provider-key-google": "g1", "authorization": "Bearer team-key"}
    assert chat(client, bearer).status_code == 200
    assert calls[-1] == ("google.test", "Bearer g1")
