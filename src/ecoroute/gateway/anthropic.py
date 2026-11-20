"""Anthropic Messages API support, so Claude Code (and any Anthropic SDK app) can route
through EcoRoute by setting ANTHROPIC_BASE_URL to the gateway.

    POST /v1/messages               route, then forward to the chosen model (streaming works)
    POST /v1/messages/count_tokens  forwarded unchanged

Only models whose provider declares `anthropic_base_url` in providers.yaml can be chosen
here: those accept this request format as is, so tool calls, thinking blocks and prompt
caching pass through untouched. Translating an agent's tool calls to another provider's
format is lossy, so it is not done.

Per request:
1. Privacy check over the user's work that will leave the building: what they typed,
   tool inputs and tool results (the client's own system prompt is skipped, see
   scan_blocks). Each block's result is cached by its hash, so a long agent conversation
   is scanned once, not on every turn. Spans no reachable model is cleared for are
   redacted and restored in the answer (redaction.py).
2. The graph scores the last thing the human typed (Claude Code's own reminders are
   removed first). An agent's follow-up turns (tool results only) share that prompt, so
   the whole task stays on one model and its prompt cache keeps working.
3. When the client asked for the cheapest model on offer (Claude Code's background tasks
   ask for Haiku), that request is kept on it; otherwise the graph chooses freely.
4. If the chosen model rejects the request (400, for example a thinking setting it does
   not support), it is retried once without those settings, then on the model the client
   asked for. Other errors go back unchanged so the client's own retry logic works.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from ecoroute.confidentiality import Classification, Detector, Level, NERLayer
from ecoroute.gateway.redaction import (
    PLACEHOLDER,
    Redactor,
    is_reminder,
    restore_message,
    restore_stream,
)
from ecoroute.routing import NoAllowedModel

log = logging.getLogger("ecoroute.gateway")

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
# Settings some models refuse; dropped on the retry after a 400.
_OPTIONAL_FIELDS = ("thinking", "output_config", "context_management", "effort")
# Request headers passed on to the provider.
_FORWARD = ("anthropic-version", "anthropic-beta", "x-api-key", "authorization")


def _blocks(content: Any) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, Mapping)]


def _block_text(block: Mapping[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "")
    if kind == "tool_result":
        return "\n".join(_block_text(b) for b in _blocks(block.get("content")))
    if kind == "tool_use":
        return json.dumps(block.get("input", {}), ensure_ascii=False)
    if kind == "document":
        src = block.get("source") or {}
        return src.get("data", "") if src.get("type") == "text" else ""
    return ""


def human_prompt(body: Mapping[str, Any]) -> str:
    """The last thing the human typed: the newest user message with text of its own (not
    only tool results), without Claude Code's <system-reminder> blocks."""
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        texts = [b.get("text", "") for b in _blocks(msg.get("content")) if b.get("type") == "text"]
        text = _REMINDER.sub("", "\n".join(texts)).strip()
        if text:
            return text
    return ""


def scan_blocks(body: Mapping[str, Any], include_system: bool = False) -> list[tuple[str, bool]]:
    """The text the request sends out that came from the user's work, one entry per block:
    what they typed, tool inputs and tool results (the files the agent read).

    The system prompt and <system-reminder> blocks are the client's own instructions,
    the same on every request, and are skipped unless include_system: they hold session
    IDs and paths that look like secrets and would otherwise block every request.

    Each entry is (text, typed): typed is True for what the human wrote (user text
    blocks), False for the agent's work (tool calls, tool results, model answers)."""
    out = [(_block_text(b), False) for b in _blocks(body.get("system"))] if include_system else []
    for msg in body.get("messages") or []:
        for b in _blocks(msg.get("content")):
            t = _block_text(b)
            if include_system or not (b.get("type") == "text" and is_reminder(t)):
                out.append((t, msg.get("role") == "user" and b.get("type") == "text"))
    return [(t, typed) for t, typed in out if t.strip()]


class ScanCache:
    """Privacy results per block of text, keyed by hash (the text itself is not kept).

    detector checks what the human typed; work_detector (default: the same) checks tool
    calls, tool results and model answers."""

    def __init__(self, detector, work_detector=None, size: int = 20_000) -> None:
        self.detector, self.size = detector, size
        self.work_detector = work_detector or detector
        self._hits: OrderedDict[bytes, Classification] = OrderedDict()

    def one(self, text: str, typed: bool = True) -> Classification:
        key = hashlib.sha256(text.encode()).digest() + (b"t" if typed else b"w")
        if key in self._hits:
            self._hits.move_to_end(key)
            return self._hits[key]
        # Our own placeholders (after redaction) are blanked, so they are not taken for
        # names or keys; spaces keep every other span where it was.
        masked = PLACEHOLDER.sub(lambda m: " " * len(m.group()), text)
        detector = self.detector if typed else self.work_detector
        result = detector.classify(masked, None)
        self._hits[key] = result
        if len(self._hits) > self.size:
            self._hits.popitem(last=False)
        return result

    def classify(self, blocks: list[tuple[str, bool]], context: Mapping | None) -> Classification:
        parts = [self.one(t, typed) for t, typed in blocks]
        if context:  # caller labels (no text) are checked on their own, never cached
            parts.append(self.detector.classify("", context))
        if not parts:
            return self.detector.classify("", None)
        findings = [f for p in parts for f in p.findings]
        return Classification(max(p.level for p in parts), findings)


def messages_models(
    catalog: list[Mapping[str, Any]], providers: Mapping[str, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Catalog models that accept Anthropic Messages requests."""
    return [
        m for m in catalog if (providers.get(m.get("provider", "")) or {}).get("anthropic_base_url")
    ]


def match_requested(requested: str, models: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The catalog model a client asked for, by name, provider ID, or family
    (claude-haiku-4-5-20251001 -> the catalog's haiku)."""
    if not requested:
        return None
    for m in models:
        if requested in (m["name"], m.get("api_id")):
            return m
    for family in ("haiku", "sonnet", "opus"):
        if family in requested:
            same = [m for m in models if family in m["name"]]
            if same:
                return max(same, key=lambda m: m["name"])
    return None


def _price(m: Mapping[str, Any]) -> float:
    return float(m.get("price_out_per_mtok") or 0.0)


def strip_optional(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in _OPTIONAL_FIELDS}


class AnthropicBackend:
    """Sends Messages requests to the chosen provider.

    The caller's own key (x-api-key or Authorization, as Claude Code sends it) is passed
    through. Without one, the provider's key from the server environment is used, unless
    the gateway is hosted for others (caller_keys_only), where callers always pay with
    their own key and the server's is never used.
    """

    def __init__(
        self,
        providers: Mapping[str, Mapping[str, Any]],
        env: Mapping[str, str] | None = None,
        caller_keys_only: bool = False,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.providers = providers
        self.env = os.environ if env is None else env
        self.caller_keys_only = caller_keys_only
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    def _headers(self, provider: Mapping[str, Any], incoming: Mapping[str, str]) -> dict:
        h = {"content-type": "application/json"}
        h.update({k: incoming[k] for k in _FORWARD if k in incoming})
        h.setdefault("anthropic-version", "2023-06-01")
        if "x-api-key" not in h and "authorization" not in h and not self.caller_keys_only:
            key_env = provider.get("api_key_env")
            if key_env and self.env.get(key_env):
                h["x-api-key"] = self.env[key_env]
        return h

    async def send(
        self,
        model: Mapping[str, Any],
        body: dict,
        incoming: Mapping[str, str],
        path: str = "/v1/messages",
        drop_beta: bool = False,
    ) -> httpx.Response:
        """Open the request (streamed); the caller reads or closes the response."""
        p = self.providers[model["provider"]]
        headers = self._headers(p, incoming)
        if drop_beta:
            headers.pop("anthropic-beta", None)
        url = p["anthropic_base_url"].rstrip("/") + path
        payload = {**body, "model": model["api_id"]}
        req = self.client.build_request("POST", url, headers=headers, json=payload)
        return await self.client.send(req, stream=True)


class EchoMessagesBackend:
    """Dry run: answers in Messages format with the model that would have been called."""

    def _text(self, model: Mapping[str, Any]) -> str:
        return f"[dry run] {model['name']} would answer this."

    async def send(self, model, body, incoming, path="/v1/messages", drop_beta=False):
        if path.endswith("count_tokens"):
            n = sum(len(t) for t, _ in scan_blocks(body, include_system=True)) // 4
            return httpx.Response(200, json={"input_tokens": max(1, n)})
        text = self._text(model)
        msg = {
            "id": "msg_ecoroute_dry_run",
            "type": "message",
            "role": "assistant",
            "model": model["name"],
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=msg)
        start = {**msg, "content": [], "stop_reason": None}
        events = [
            ("message_start", {"type": "message_start", "message": start}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"}),
        ]  # fmt: skip
        sse = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )


async def relay(response: httpx.Response) -> AsyncIterator[bytes]:
    """Pass the provider's bytes on as they arrive (no buffering), then close."""
    try:
        if hasattr(response, "_content"):  # already in memory (dry run, tests)
            yield response.content
        else:
            async for chunk in response.aiter_raw():
                yield chunk
    finally:
        await response.aclose()


# Provider response headers worth passing on (retry hints, rate limits, request IDs).
def passthrough_headers(response: httpx.Response) -> dict[str, str]:
    keep = {}
    for k, v in response.headers.items():
        k = k.lower()
        if k in ("retry-after", "x-should-retry", "request-id") or k.startswith(
            "anthropic-ratelimit"
        ):
            keep[k] = v
    return keep


def _ascii(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v.encode("ascii", "replace").decode() for k, v in headers.items()}


def register_messages(
    app,
    eco,
    backend,
    providers: Mapping[str, Mapping[str, Any]],
    context_of,
    redact: bool = True,
    include_system: bool = False,
    pii_on_tools: bool = False,
):
    """Add /v1/messages and /v1/messages/count_tokens to the gateway app.

    redact: when the request holds text no reachable model is cleared for, replace those
    spans with placeholders and restore them in the answer (see redaction.py) instead of
    refusing the request. Caller labels (X-EcoRoute-Min-Level) are never redacted away.

    pii_on_tools: also run the PII model over tool calls and tool results. Off by default:
    the rules still catch keys, cards and IDs in every file the agent reads, while the PII
    model is slow on whole files (seconds per file on a CPU) and mistakes code for names
    (it tags model IDs such as claude-opus-5-5 as usernames)."""
    pool = messages_models(eco.catalog, providers)
    by_name = {m["name"]: m for m in pool}
    det = eco.router.detector
    work = det
    if not pii_on_tools:
        work = Detector(
            [layer for layer in det.layers if not isinstance(layer, NERLayer)],
            floor=det.floor, weak_kinds=det.weak_kinds, stop_at_restricted=det.stop_at_restricted,
        )  # fmt: skip
    scans = ScanCache(det, work)
    decisions: OrderedDict[tuple, Any] = OrderedDict()
    reason = "the models that accept Anthropic Messages requests (Claude Code's format)"

    def choose(body: dict, request: Request):
        if not pool:
            raise HTTPException(503, "no model with an anthropic_base_url provider is deployed")
        requested = str(body.get("model") or "")
        asked = match_requested(requested, pool)
        names, why = [m["name"] for m in pool], reason
        cheapest = min(_price(m) for m in pool)
        if asked is not None and _price(asked) <= cheapest:
            names = [asked["name"]]
            why = f"the client asked for the cheapest model, {asked['name']}, and keeps it"
        context = context_of(request)
        conf = scans.classify(scan_blocks(body, include_system), context)
        prompt = human_prompt(body) or "\n".join(t for t, _ in scan_blocks(body)[-1:])
        clearance = max(Level.parse(m.get("clearance", "public")) for m in pool)
        mapping, kinds = {}, {}
        if redact and conf.level > clearance:
            r = Redactor(scans.one, clearance)
            # A placeholder changes the text around it, and the PII model can then see
            # something new nearby, so repeat until nothing is left (a few rounds at most).
            for _ in range(4):
                body = r.body(body)
                conf = scans.classify(scan_blocks(body, include_system), context)
                if conf.level <= clearance:
                    break
            mapping, kinds = r.mapping, r.kinds
        policy = request.headers.get("x-ecoroute-policy")
        key = (hashlib.sha256(prompt.encode()).digest(), conf.level, tuple(names), policy)
        d = decisions.get(key)
        if d is None:
            try:
                d = eco.route(
                    prompt, context=context, policy=policy,
                    out_tokens=int(body.get("max_tokens") or 500),
                    classification=conf, models=names, models_reason=why,
                )  # fmt: skip
            except NoAllowedModel as e:
                raise HTTPException(403, f"no model reachable here is cleared for this: {e}") from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            decisions[key] = d
            if len(decisions) > 2000:
                decisions.popitem(last=False)
        else:
            decisions.move_to_end(key)
        allowed = {c.name for c in d.candidates if c.allowed}
        asked = asked if asked is not None and asked["name"] in allowed else None
        return d, asked, body, mapping, kinds

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        incoming = {k.lower(): v for k, v in request.headers.items()}
        d, asked, body, mapping, kinds = await run_in_threadpool(choose, body, request)
        chosen = by_name[d.model]
        headers = d.headers()
        if kinds:
            headers["X-EcoRoute-Redacted"] = ", ".join(f"{k} x{n}" for k, n in kinds.items())
        attempts = [(chosen, body, False)]
        if asked is None or asked["name"] != chosen["name"]:
            # The client built this request for another model: if ours refuses a setting,
            # drop the optional ones, then fall back to what the client asked for.
            attempts.append((chosen, strip_optional(body), True))
            if asked is not None:
                attempts.append((asked, body, False))
        for i, (model, payload, drop_beta) in enumerate(attempts):
            r = await backend.send(model, payload, incoming, drop_beta=drop_beta)
            if r.status_code != 400 or i == len(attempts) - 1:
                break
            detail = (await r.aread())[:300].decode("utf-8", "replace")
            await r.aclose()
            log.warning("%s refused the request (400 %s); retrying", model["name"], detail)
        if model is not chosen or i > 0:
            headers["X-EcoRoute-Model"] = model["name"]
            headers["X-EcoRoute-Reason"] = (
                f"{chosen['name']} refused the request, sent to {model['name']}"
                + (" without optional settings" if drop_beta else "")
            )
        log.info(
            "messages route %s (asked for %s) level=%s", model["name"], body.get("model"), d.level
        )
        out = {**_ascii(headers), **passthrough_headers(r)}
        ctype = r.headers.get("content-type", "application/json")
        if body.get("stream") and r.status_code == 200:
            stream = restore_stream(relay(r), mapping) if mapping else relay(r)
            return StreamingResponse(
                stream, status_code=200, media_type="text/event-stream", headers=out
            )
        content = await r.aread()
        await r.aclose()
        if mapping and r.status_code == 200:
            try:
                content = json.dumps(restore_message(json.loads(content), mapping)).encode()
            except (ValueError, AttributeError):
                pass
        return Response(content, status_code=r.status_code, media_type=ctype, headers=out)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        body = await request.json()
        if not pool:
            raise HTTPException(503, "no model with an anthropic_base_url provider is deployed")
        model = match_requested(str(body.get("model") or ""), pool) or min(pool, key=_price)
        incoming = {k.lower(): v for k, v in request.headers.items()}
        r = await backend.send(model, body, incoming, path="/v1/messages/count_tokens")
        content = await r.aread()
        await r.aclose()
        return Response(
            content, status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )  # fmt: skip

    return pool
