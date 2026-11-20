"""Anthropic Messages endpoint (/v1/messages), the path Claude Code uses."""

import json

import httpx
import numpy as np
from fastapi.testclient import TestClient

from ecoroute.gateway import AnthropicBackend, EchoBackend, EchoMessagesBackend, create_app
from ecoroute.gateway.anthropic import human_prompt, match_requested, scan_blocks
from ecoroute.predictors import ModelMeanPredictor
from ecoroute.service import EcoRoute

AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
PROVIDERS = {
    "anthropic": {"anthropic_base_url": "https://anthropic.test", "api_key_env": "ANTHROPIC_KEY"},
    "local": {"base_url": "http://local.test/v1", "api_key_env": None},
}
CATALOG = [
    {"name": "local-8b", "provider": "local", "api_id": "llama", "clearance": "restricted",
     "anchor": {"model": "a"}},
    {"name": "claude-haiku-4-5", "provider": "anthropic", "api_id": "claude-haiku-4-5",
     "clearance": "internal", "price_out_per_mtok": 5.0, "anchor": {"model": "b"}},
    {"name": "claude-opus-5-5", "provider": "anthropic", "api_id": "claude-opus-5-5",
     "clearance": "internal", "price_out_per_mtok": 25.0, "anchor": {"model": "c"}},
]  # fmt: skip


class Upstream:
    """Fake api.anthropic.com: records requests, can refuse some models with a 400."""

    def __init__(self, refuse=()):
        self.calls, self.refuse = [], dict(refuse)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.url.path, body, dict(request.headers)))
        if request.url.path.endswith("count_tokens"):
            return httpx.Response(200, json={"input_tokens": 7})
        rule = self.refuse.get(body["model"])
        if rule and rule(body, request.headers):
            return httpx.Response(400, json={"type": "error", "error": {"message": "nope"}})
        if body.get("stream"):
            sse = b'event: message_start\ndata: {"type":"message_start"}\n\n'
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={"id": "msg_1", "type": "message", "model": body["model"], "content": []},
            headers={"retry-after": "3", "anthropic-ratelimit-requests-remaining": "9"},
        )


def make_client(upstream, p=(0.99, 0.95, 0.97), env=None, caller_keys_only=False):
    base = ModelMeanPredictor().fit(np.zeros((2, 3)), np.array([p, p]), ["a", "b", "c"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = lambda texts: np.zeros((len(texts), 3), np.float32)
    backend = AnthropicBackend(
        PROVIDERS,
        env=env or {"ANTHROPIC_KEY": "server-key"},
        caller_keys_only=caller_keys_only,
        transport=httpx.MockTransport(upstream),
    )
    return TestClient(create_app(eco, EchoBackend(), messages_backend=backend, providers=PROVIDERS))


def send(client, content, model="claude-opus-5-5", headers=None, **kw):
    body = {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": content}]}
    return client.post("/v1/messages", json={**body, **kw}, headers=headers or {})


def test_routes_to_cheapest_qualifying_claude_model_never_to_non_messages_model():
    up = Upstream()
    r = send(make_client(up), "what is 2+2?")
    assert r.status_code == 200
    # local-8b is cheaper and most likely to succeed, but cannot take Messages requests.
    assert up.calls[0][1]["model"] == "claude-haiku-4-5"
    assert r.headers["x-ecoroute-model"] == "claude-haiku-4-5"
    assert r.headers["retry-after"] == "3"
    assert r.headers["anthropic-ratelimit-requests-remaining"] == "9"


def test_hard_prompt_goes_to_the_bigger_model():
    up = Upstream()
    send(make_client(up, p=(0.5, 0.6, 0.95)), "prove it")
    assert up.calls[0][1]["model"] == "claude-opus-5-5"


def test_request_for_the_cheapest_model_stays_on_it():
    up = Upstream()
    send(make_client(up, p=(0.5, 0.6, 0.95)), "title this chat", model="claude-haiku-4-5-2025")
    assert up.calls[0][1]["model"] == "claude-haiku-4-5"


def test_caller_key_and_headers_are_forwarded_and_win_over_server_key():
    up = Upstream()
    hdrs = {"x-api-key": "callers-own", "anthropic-beta": "x-1", "anthropic-version": "2023-06-01"}
    send(make_client(up), "hi", headers=hdrs)
    sent = up.calls[0][2]
    assert sent["x-api-key"] == "callers-own"
    assert sent["anthropic-beta"] == "x-1"


def test_server_key_used_only_when_not_hosted():
    up = Upstream()
    send(make_client(up), "hi")
    assert up.calls[0][2]["x-api-key"] == "server-key"
    up = Upstream()
    send(make_client(up, caller_keys_only=True), "hi")
    assert "x-api-key" not in up.calls[0][2]


def _secret_conversation():
    return [
        {"role": "user", "content": "read my config"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read",
                                           "input": {"path": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                      "content": f"aws_key = {AWS}"}]},
    ]  # fmt: skip


def test_secret_in_tool_result_is_redacted_before_it_leaves_and_restored_after():
    class Echo(Upstream):
        def __call__(self, request):
            body = json.loads(request.content)
            self.calls.append((request.url.path, body, dict(request.headers)))
            seen = body["messages"][-1]["content"][0]["content"]
            tag = seen.split("= ")[1]
            return httpx.Response(200, json={"type": "message", "content": [
                {"type": "text", "text": f"the key is {tag}"},
                {"type": "tool_use", "id": "w", "name": "Write", "input": {"content": tag}}]})  # fmt: skip

    up = Echo()
    body = {"model": "claude-opus-5-5", "max_tokens": 10, "messages": _secret_conversation()}
    r = make_client(up).post("/v1/messages", json=body)
    assert r.status_code == 200
    assert AWS not in json.dumps(up.calls[0][1])  # never left the gateway
    assert "AWS" in r.headers["x-ecoroute-redacted"].upper()
    out = r.json()["content"]
    assert out[0]["text"] == f"the key is {AWS}" and out[1]["input"]["content"] == AWS


def test_no_redact_refuses_and_caller_label_is_never_redacted_away():
    base = ModelMeanPredictor().fit(np.zeros((2, 3)), np.ones((2, 3)), ["a", "b", "c"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = lambda texts: np.zeros((len(texts), 3), np.float32)
    up = Upstream()
    backend = AnthropicBackend(PROVIDERS, env={}, transport=httpx.MockTransport(up))
    app = create_app(eco, EchoBackend(), messages_backend=backend, providers=PROVIDERS,
                     redact=False)  # fmt: skip
    body = {"model": "claude-opus-5-5", "max_tokens": 10, "messages": _secret_conversation()}
    assert TestClient(app).post("/v1/messages", json=body).status_code == 403
    r = send(make_client(up), "hello", headers={"x-ecoroute-min-level": "restricted"})
    assert r.status_code == 403 and up.calls == []


def test_stream_restores_placeholder_split_across_deltas():
    import asyncio

    from ecoroute.gateway.redaction import placeholder, restore_stream

    tag = placeholder("aws_access_key", AWS)
    events = [
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "key " + tag[:7]}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": tag[7:] + " ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"v": "' + tag[:3]}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": tag[3:] + '"}'}},
        {"type": "content_block_stop", "index": 1},
    ]  # fmt: skip
    raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    async def chunks():
        for i in range(0, len(raw), 17):  # arbitrary network chunking
            yield raw[i : i + 17]

    async def collect():
        return b"".join([c async for c in restore_stream(chunks(), {tag: AWS})])

    out = asyncio.run(collect()).decode()
    text, js = "", ""
    for block in out.split("\n\n"):
        if "data:" not in block:
            continue
        d = json.loads(block.split("data:", 1)[1])
        if d["type"] == "content_block_delta":
            text += d["delta"].get("text", "")
            js += d["delta"].get("partial_json", "")
    assert text == f"key {AWS} ok" and json.loads(js) == {"v": AWS}


def test_refused_setting_is_retried_without_it_then_on_requested_model():
    up = Upstream(refuse={"claude-haiku-4-5": lambda body, h: "thinking" in body})
    r = send(make_client(up), "hi", thinking={"type": "adaptive"})
    assert r.status_code == 200
    assert [c[1]["model"] for c in up.calls] == ["claude-haiku-4-5", "claude-haiku-4-5"]
    assert "thinking" not in up.calls[1][1]
    assert "without optional settings" in r.headers["x-ecoroute-reason"]

    up = Upstream(refuse={"claude-haiku-4-5": lambda body, h: True})
    r = send(make_client(up), "hi", thinking={"type": "adaptive"})
    assert r.status_code == 200
    assert [c[1]["model"] for c in up.calls][-1] == "claude-opus-5-5"
    assert "thinking" in up.calls[-1][1]


def test_other_errors_pass_through_unchanged():
    def down(request):
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})

    r = send(make_client(down), "hi")
    assert r.status_code == 529 and "overloaded_error" in r.text


def test_streaming_passes_events_through():
    up = Upstream()
    r = send(make_client(up), "hi", stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "message_start" in r.text


def test_count_tokens_forwarded():
    up = Upstream()
    r = make_client(up).post(
        "/v1/messages/count_tokens",
        json={"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.json() == {"input_tokens": 7} and up.calls[0][0] == "/v1/messages/count_tokens"


def test_route_endpoint_can_limit_to_messages_models():
    client = make_client(Upstream())
    body = {"messages": [{"role": "user", "content": "what is 2+2?"}]}
    assert client.post("/route", json=body).json()["model"] == "local-8b"
    r = client.post("/route", json=body, headers={"x-ecoroute-api": "messages"})
    assert r.json()["model"] == "claude-haiku-4-5"


def test_dry_run_backend_answers_in_messages_format():
    base = ModelMeanPredictor().fit(np.zeros((2, 3)), np.ones((2, 3)), ["a", "b", "c"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = lambda texts: np.zeros((len(texts), 3), np.float32)
    app = create_app(
        eco, EchoBackend(), messages_backend=EchoMessagesBackend(), providers=PROVIDERS
    )
    r = send(TestClient(app), "hi")
    assert r.json()["type"] == "message" and "would answer" in r.json()["content"][0]["text"]
    r = send(TestClient(app), "hi", stream=True)
    assert "message_stop" in r.text


def test_human_prompt_skips_tool_results_and_reminders():
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                {"type": "text", "text": "fix the bug"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                          "content": "file text"}]},
        ],
        "system": [{"type": "text", "text": "You are Claude Code"}],
    }  # fmt: skip
    assert human_prompt(body) == "fix the bug"
    blocks = dict(scan_blocks(body))
    assert blocks == {"fix the bug": True, "ok": False, "file text": False}
    assert "You are Claude Code" in dict(scan_blocks(body, include_system=True))


def test_match_requested_by_family():
    pool = [m for m in CATALOG if m["provider"] == "anthropic"]
    assert match_requested("claude-haiku-4-5-20251001", pool)["name"] == "claude-haiku-4-5"
    assert match_requested("ecoroute/auto", pool) is None
