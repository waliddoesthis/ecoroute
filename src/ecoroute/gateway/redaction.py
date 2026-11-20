"""Redact secrets in an agent's request and put them back in the answer.

A coding agent reads files, and files hold keys, hashes and personal data. Blocking the
request would stop the agent for good, since the conversation keeps that text. Instead the
spans no reachable model is cleared for are replaced by placeholders such as
<AWS_ACCESS_KEY_3f9a1c2b> before the request leaves, and the real values are put back in
the model's answer (text and tool-call arguments), so when the agent writes a file the
real value lands on disk.

Placeholders come from a hash of the value, so the same secret gets the same placeholder
on every turn: the request prefix stays identical and the provider's prompt cache still
works. Values live only in this process's memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from ecoroute.confidentiality import Level

PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*_[0-9a-f]{8}>")
_HOLD = 64  # longest placeholder prefix kept back between stream chunks


def placeholder(kind: str, value: str) -> str:
    tag = re.sub(r"[^A-Z0-9]+", "_", kind.split(":")[0].upper()).strip("_") or "SECRET"
    return f"<{tag}_{hashlib.sha256(value.encode()).hexdigest()[:8]}>"


class Redactor:
    """classify(text, typed) -> Classification (cached); limit: the highest level allowed
    out. typed is True for what the human wrote, False for the agent's work."""

    def __init__(self, classify: Callable[[str, bool], Any], limit: Level) -> None:
        self.classify, self.limit = classify, limit
        self.mapping: dict[str, str] = {}
        self.kinds: dict[str, int] = {}

    def text(self, text: str, typed: bool = False) -> str:
        spans = [
            f
            for f in self.classify(text, typed).findings
            if f.end > f.start and f.level > self.limit
        ]
        if not spans:
            return text
        out, pos = [], 0
        for f in sorted(spans, key=lambda f: f.start):
            if f.start < pos:
                continue
            value = text[f.start : f.end]
            tag = placeholder(f.kind, value)
            self.mapping[tag] = value
            self.kinds[f.kind] = self.kinds.get(f.kind, 0) + 1
            out += [text[pos : f.start], tag]
            pos = f.end
        out.append(text[pos:])
        return "".join(out)

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        return v

    def block(self, b: Any, typed: bool = False) -> Any:
        if not isinstance(b, dict):
            return b
        kind = b.get("type")
        if kind == "text" and not is_reminder(b.get("text", "")):
            return {**b, "text": self.text(b.get("text", ""), typed)}
        if kind == "tool_use":
            return {**b, "input": self.value(b.get("input", {}))}
        if kind == "tool_result":
            c = b.get("content")
            c = self.text(c) if isinstance(c, str) else [self.block(x) for x in c or []]
            return {**b, "content": c}
        return b  # thinking blocks are signed and must stay byte-identical

    def body(self, body: Mapping[str, Any]) -> dict:
        msgs = []
        for m in body.get("messages") or []:
            c = m.get("content")
            typed = m.get("role") == "user"
            if isinstance(c, str):
                c = self.text(c, typed)
            else:
                c = [self.block(b, typed) for b in c or []]
            msgs.append({**m, "content": c})
        return {**body, "messages": msgs}


def is_reminder(text: str) -> bool:
    return text.lstrip().startswith("<system-reminder>")


def restore(text: str, mapping: Mapping[str, str], in_json: bool = False) -> str:
    def put(m: re.Match) -> str:
        v = mapping.get(m.group())
        if v is None:
            return m.group()
        return json.dumps(v)[1:-1] if in_json else v

    return PLACEHOLDER.sub(put, text)


def restore_message(msg: dict, mapping: Mapping[str, str]) -> dict:
    def walk(v):
        if isinstance(v, str):
            return restore(v, mapping)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    content = []
    for b in msg.get("content") or []:
        if b.get("type") == "text":
            b = {**b, "text": restore(b.get("text", ""), mapping)}
        elif b.get("type") == "tool_use":
            b = {**b, "input": walk(b.get("input", {}))}
        content.append(b)
    return {**msg, "content": content}


def _split_hold(s: str) -> tuple[str, str]:
    """(safe to send now, possible start of a placeholder to keep for the next chunk)."""
    i = s.rfind("<")
    if i == -1 or ">" in s[i:] or len(s) - i > _HOLD:
        return s, ""
    return s[:i], s[i:]


async def restore_stream(
    chunks: AsyncIterator[bytes], mapping: Mapping[str, str]
) -> AsyncIterator[bytes]:
    """Restore placeholders in a Messages SSE stream, including ones split across deltas."""
    buf = b""
    pending: dict[int, tuple[str, str]] = {}  # block index -> (delta type, held text)

    def event(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    def flush(index: int) -> bytes:
        kind, held = pending.pop(index, ("", ""))
        if not held:
            return b""
        field = "text" if kind == "text_delta" else "partial_json"
        text = restore(held, mapping, in_json=kind == "input_json_delta")
        delta = {"type": kind, field: text}
        return event("content_block_delta", {"type": "content_block_delta", "index": index,
                                             "delta": delta})  # fmt: skip

    async for chunk in chunks:
        buf += chunk
        while b"\n\n" in buf:
            raw, buf = buf.split(b"\n\n", 1)
            lines = raw.decode("utf-8", "replace").split("\n")
            data_lines = [ln[5:].lstrip() for ln in lines if ln.startswith("data:")]
            name = next((ln[6:].strip() for ln in lines if ln.startswith("event:")), "")
            try:
                data = json.loads("\n".join(data_lines)) if data_lines else None
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict):
                yield raw + b"\n\n"
                continue
            kind = data.get("type")
            if kind == "content_block_delta":
                delta = data.get("delta") or {}
                dtype = delta.get("type")
                field = {"text_delta": "text", "input_json_delta": "partial_json"}.get(dtype)
                if field is None:
                    yield raw + b"\n\n"
                    continue
                idx = data.get("index", 0)
                held = pending.get(idx, (dtype, ""))[1]
                send, keep = _split_hold(held + delta.get(field, ""))
                pending[idx] = (dtype, keep)
                send = restore(send, mapping, in_json=dtype == "input_json_delta")
                yield event(name or kind, {**data, "delta": {**delta, field: send}})
                continue
            if kind == "content_block_stop":
                yield flush(data.get("index", 0))
            yield raw + b"\n\n"
    for idx in list(pending):
        yield flush(idx)
    if buf:
        yield buf
