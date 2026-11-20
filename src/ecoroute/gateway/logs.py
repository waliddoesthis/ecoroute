"""Decision log and feedback, the raw material for retraining on your own work.

With `serve.py --log-dir DIR` the gateway appends one line per routed request to
DIR/decisions.jsonl and returns its id in the X-EcoRoute-Request-Id header. Tell it how the
answer went with

    curl -X POST localhost:8080/feedback -d '{"id": "<request id>", "success": true}'

(`success` may also be a score from 0 to 1; add `"model"` when you graded another model's
answer to the same prompt). Lines go to DIR/feedback.jsonl, and scripts/retrain.py joins
the two files.

The log holds prompt text, so it is off by default. Secrets the rules find (keys, card
numbers, IDs) are replaced by placeholders before a prompt is written.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from ecoroute.confidentiality import Level, RulesLayer
from ecoroute.confidentiality.redact import redact


class DecisionLog:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rules = RulesLayer()
        self._lock = threading.Lock()

    def _append(self, name: str, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(line)

    def decision(self, prompt: str, decision, api: str) -> str:
        rid = uuid.uuid4().hex
        found = [f for f in self.rules.scan(prompt) if f.level > Level.INTERNAL]
        safe = redact(prompt, found)[0] if found else prompt
        self._append(
            "decisions.jsonl",
            {
                "id": rid,
                "ts": time.time(),
                "api": api,
                "prompt": safe,
                "model": decision.model,
                "level": str(decision.level),
                "tau": decision.policy.tau,
                "p_success": decision.chosen.p_success,
            },
        )
        return rid

    def feedback(self, record: dict) -> None:
        if not record.get("id"):
            raise ValueError("feedback needs the request id")
        score = record.get("success", record.get("score"))
        if isinstance(score, bool):
            score = float(score)
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("success must be true/false or a score from 0 to 1")
        out = {"id": record["id"], "ts": time.time(), "success": float(score)}
        if record.get("model"):
            out["model"] = str(record["model"])
        self._append("feedback.jsonl", out)


def read_logged_outcomes(directory: str | Path) -> list[dict]:
    """(prompt, model, success) records from a gateway log directory: each feedback line
    joined with its decision (the model is the one routed to unless the feedback names
    another)."""
    d = Path(directory)
    decisions = {}
    with open(d / "decisions.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                decisions[r["id"]] = r
    out = []
    path = d / "feedback.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            fb = json.loads(line)
            dec = decisions.get(fb["id"])
            if dec is not None:
                out.append(
                    {
                        "prompt": dec["prompt"],
                        "model": fb.get("model", dec["model"]),
                        "success": fb["success"],
                    }
                )
    return out
