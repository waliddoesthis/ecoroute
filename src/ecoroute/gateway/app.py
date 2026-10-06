"""OpenAI-compatible gateway: apps change base_url and use model="ecoroute/auto".

    POST /v1/chat/completions  route, then forward to the chosen provider (streaming works)
    POST /route                decision and explanation only, no model is called
    GET  /v1/models            the models this gateway can route to

Request headers the caller may set:
    X-EcoRoute-Sensitivity   the data's sensitivity label (e.g. "Confidential")
    X-EcoRoute-Min-Level     public / internal / confidential / restricted
    X-EcoRoute-Policy        eco / balanced / quality

Asking for a specific model instead of ecoroute/auto skips the quality ranking but never
the confidentiality check: an uncleared model is refused with 403.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ecoroute.gateway.backends import Backend
from ecoroute.routing import NoAllowedModel

log = logging.getLogger("ecoroute.gateway")
AUTO = "ecoroute/auto"


def _text(content: Any) -> str:
    """Message content as plain text (handles OpenAI's list-of-parts form)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def split_messages(messages: list[Mapping[str, Any]]) -> tuple[str, str]:
    """(last user message, whole conversation). The predictor scores the question being
    asked; the confidentiality check reads everything that will leave the building."""
    texts = [_text(m.get("content")) for m in messages]
    users = [_text(m.get("content")) for m in messages if m.get("role") == "user"]
    return (users[-1] if users else "\n".join(texts)), "\n".join(texts)


def _context(request: Request) -> dict:
    ctx = {}
    if label := request.headers.get("x-ecoroute-sensitivity"):
        ctx["sensitivity_label"] = label
    if level := request.headers.get("x-ecoroute-min-level"):
        ctx["min_level"] = level
    return ctx


def create_app(eco, backend: Backend) -> FastAPI:
    """eco: an EcoRoute (or anything with .route(), .router and .catalog) whose catalog is
    already limited to deployable models."""
    app = FastAPI(title="EcoRoute gateway")
    by_name = {m["name"]: m for m in eco.catalog}

    def decide(body: dict, request: Request):
        messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
        prompt, conversation = split_messages(messages)
        try:
            return eco.route(
                prompt,
                context=_context(request),
                policy=request.headers.get("x-ecoroute-policy"),
                out_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 500),
                scan_text=conversation,
            )
        except NoAllowedModel as e:
            raise HTTPException(403, f"no deployed model is cleared for this prompt: {e}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/v1/models")
    def models():
        data = [{"id": AUTO, "object": "model", "owned_by": "ecoroute"}]
        data += [
            {"id": n, "object": "model", "owned_by": m["provider"]} for n, m in by_name.items()
        ]
        return {"object": "list", "data": data}

    @app.post("/route")
    async def route(request: Request):
        d = decide(await request.json(), request)
        return JSONResponse(
            {
                "model": d.model,
                "level": str(d.level),
                "difficulty": d.difficulty,
                "below_quality_floor": d.below_floor,
                "explanation": d.steps,
                "candidates": [
                    {
                        "name": c.name,
                        "p_success": c.p_success,
                        "cost_usd": c.cost_usd,
                        "energy_wh": c.energy_wh,
                        "allowed": c.allowed,
                        "qualifies": c.qualifies,
                        "excluded_because": c.excluded_because,
                    }
                    for c in d.candidates
                ],
            },
            headers=d.headers(),
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        d = decide(body, request)
        headers = d.headers()
        target = body.get("model", AUTO)
        if target not in (AUTO, None, ""):
            if target not in by_name:
                raise HTTPException(404, f"unknown model {target!r}")
            cand = next(c for c in d.candidates if c.name == target)
            if not cand.allowed:
                raise HTTPException(
                    403, f"{target} is not cleared for {d.level} content ({cand.excluded_because})"
                )
            headers["X-EcoRoute-Model"] = target
            headers["X-EcoRoute-Reason"] = (
                f"caller asked for {target}; confidentiality check passed"
            )
        chosen = by_name[headers["X-EcoRoute-Model"]]
        # Headers carry only ASCII-safe text.
        headers = {k: v.encode("ascii", "replace").decode() for k, v in headers.items()}
        log.info("route %s level=%s", chosen["name"], d.level)
        if body.get("stream"):
            return StreamingResponse(
                backend.stream(chosen, body), media_type="text/event-stream", headers=headers
            )
        try:
            result = await backend.complete(chosen, body)
        except Exception as e:  # noqa: BLE001
            fallback = _fallback(eco, d, chosen["name"])
            if fallback is None:
                raise HTTPException(502, f"{chosen['name']} failed: {e}") from e
            log.warning("%s failed (%s); falling back to %s", chosen["name"], e, fallback["name"])
            result = await backend.complete(fallback, body)
            headers["X-EcoRoute-Model"] = fallback["name"]
            headers["X-EcoRoute-Reason"] = (
                f"{chosen['name']} failed; fell back to {fallback['name']}"
            )
        return JSONResponse(result, headers=headers)

    return app


def _fallback(eco, decision, failed: str):
    """Most likely allowed model other than the one that failed (never an uncleared one)."""
    options = [c for c in decision.candidates if c.allowed and c.name != failed]
    if not options:
        return None
    best = max(options, key=lambda c: c.p_success or 0)
    return next(m for m in eco.catalog if m["name"] == best.name)
