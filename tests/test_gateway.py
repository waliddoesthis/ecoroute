import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecoroute.gateway import create_app, deployable
from ecoroute.gateway.app import split_messages
from ecoroute.predictors import ModelMeanPredictor
from ecoroute.service import EcoRoute

AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
CATALOG = [
    {"name": "local", "provider": "vllm", "api_id": "qwen", "clearance": "restricted",
     "anchor": {"model": "a"}},
    {"name": "cheap", "provider": "cloud", "api_id": "c-1", "clearance": "internal",
     "price_out_per_mtok": 1.0, "anchor": {"model": "b"}},
    {"name": "premium", "provider": "cloud", "api_id": "p-1", "clearance": "internal",
     "price_out_per_mtok": 20.0, "anchor": {"model": "c"}},
]  # fmt: skip


class FakeBackend:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    async def complete(self, model, body):
        self.calls.append(model["name"])
        if model["name"] in self.fail:
            raise RuntimeError("provider down")
        return {
            "choices": [{"message": {"role": "assistant", "content": f"hi from {model['name']}"}}]
        }

    async def stream(self, model, body):
        self.calls.append(model["name"])
        yield b"data: hello\n\n"
        yield b"data: [DONE]\n\n"


def make_client(backend, p=(0.6, 0.8, 0.95)):
    base = ModelMeanPredictor().fit(np.zeros((2, 3)), np.array([p, p]), ["a", "b", "c"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = lambda texts: np.zeros((len(texts), 3), np.float32)
    return TestClient(create_app(eco, backend))


def chat(client, content, model="ecoroute/auto", headers=None, **kw):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **kw}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def test_auto_picks_cheapest_qualifying_and_explains():
    backend = FakeBackend()
    r = chat(make_client(backend), "what is 2+2?")
    assert r.status_code == 200 and backend.calls == ["cheap"]
    assert r.headers["x-ecoroute-model"] == "cheap"
    assert r.headers["x-ecoroute-level"] == "internal"
    assert "hi from cheap" in r.text


def test_secret_in_earlier_message_keeps_it_local():
    backend = FakeBackend()
    body = {
        "model": "ecoroute/auto",
        "messages": [
            {"role": "user", "content": f"my key is {AWS}"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now write a haiku"},
        ],
    }
    r = make_client(backend).post("/v1/chat/completions", json=body)
    assert r.status_code == 200 and backend.calls == ["local"]
    assert AWS not in str(r.headers)


def test_explicit_uncleared_model_is_refused():
    backend = FakeBackend()
    r = chat(make_client(backend), f"debug {AWS}", model="premium")
    assert r.status_code == 403 and backend.calls == []


def test_caller_label_header():
    backend = FakeBackend()
    r = chat(make_client(backend), "summarise", headers={"X-EcoRoute-Sensitivity": "Restricted"})
    assert backend.calls == ["local"] and r.headers["x-ecoroute-level"] == "restricted"


def test_policy_header_moves_to_premium():
    backend = FakeBackend()
    chat(make_client(backend), "explain", headers={"X-EcoRoute-Policy": "quality"})
    assert backend.calls == ["premium"]  # only P=0.95 clears tau=0.85


def test_provider_failure_falls_back_to_an_allowed_model():
    backend = FakeBackend(fail={"cheap"})
    r = chat(make_client(backend), "hello")
    assert r.status_code == 200 and backend.calls == ["cheap", "premium"]
    assert r.headers["x-ecoroute-model"] == "premium"


def test_streaming_passes_through():
    r = chat(make_client(FakeBackend()), "hello", stream=True)
    assert r.status_code == 200 and "[DONE]" in r.text


def test_route_endpoint_calls_no_model():
    backend = FakeBackend()
    r = make_client(backend).post("/route", json={"prompt": "hello"})
    assert r.status_code == 200 and backend.calls == []
    assert r.json()["model"] == "cheap" and len(r.json()["candidates"]) == 3


@pytest.mark.parametrize(
    "env, expected",
    [({}, ["local"]), ({"CLOUD_KEY": "x"}, ["local", "cheap", "premium"])],
)
def test_deployable_needs_key_and_api_id(env, expected):
    providers = {"vllm": {"api_key_env": None}, "cloud": {"api_key_env": "CLOUD_KEY"}}
    catalog = CATALOG + [{"name": "unconfirmed", "provider": "cloud", "api_id": None}]
    assert [m["name"] for m in deployable(catalog, providers, env)] == expected


def test_split_messages_handles_content_parts():
    q, conv = split_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
    )
    assert q == "hi" and "be brief" in conv
