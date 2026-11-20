"""Retrain the router on your own graded prompts (used by scripts/retrain.py).

Records are {"prompt": str, "model": catalog name, "success": 0..1}: whether that model's
answer to that prompt was good enough. They come from your own evaluation, or from the
gateway's decision log plus /feedback (gateway/logs.py).

adapt() splits the prompts 80/20 by a hash of their text, fits on the 80% (per-model
shifts, then similar-prompt evidence, see predictors/local.py), and reports on the 20%
whether the predictions got closer to what happened (Brier score, lower is better). It
then refits on everything for deployment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ecoroute.predictors.local import LocalOutcomes, brier


def read_records(path: str | Path) -> list[dict]:
    """JSONL or CSV with prompt, model and success (true/false, yes/no or 0..1)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return [_clean(r) for r in rows]


def _clean(r: dict) -> dict:
    s = r.get("success", r.get("score"))
    if isinstance(s, str):
        s = {"true": 1, "yes": 1, "false": 0, "no": 0}.get(s.strip().lower(), s)
    s = float(s)
    if not 0 <= s <= 1:
        raise ValueError(f"success must be between 0 and 1, got {s}")
    return {"prompt": str(r["prompt"]), "model": str(r["model"]), "success": s}


def _in_test(prompt: str, share: float) -> bool:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def adapt(
    eco,
    records: Iterable[dict],
    test_share: float = 0.2,
    min_per_model: int = 30,
    k: int = 16,
    strength: float = 4.0,
    min_similarity: float = 0.5,
    fit_shifts: bool = True,
) -> dict:
    """Adapt `eco` in place (eco.shifts, eco.local) and return a report."""
    records = list(records)
    known = set(eco.predictor.names)
    unknown = sorted({r["model"] for r in records} - known)
    records = [r for r in records if r["model"] in known]
    if not records:
        raise ValueError(f"no records for catalog models (unknown models: {unknown})")
    prompts = sorted({r["prompt"] for r in records})
    X_by = dict(zip(prompts, eco.embed(prompts)))
    X = np.stack([X_by[r["prompt"]] for r in records])
    models = [r["model"] for r in records]
    y = np.array([r["success"] for r in records])
    test = np.array([_in_test(r["prompt"], test_share) for r in records])
    report = {
        "records": len(records),
        "prompts": len(prompts),
        "unknown_models_dropped": unknown,
        "per_model": {m: models.count(m) for m in sorted(set(models))},
    }
    base_shifts = {n: a.shift for n, a in eco.predictor.anchors.items()}

    def fit(rows: np.ndarray) -> tuple[dict, LocalOutcomes]:
        for n, s in base_shifts.items():
            eco.predictor.anchors[n].shift = s
        shifts = {}
        if fit_shifts:
            for m in sorted(set(models)):
                sel = rows & (np.array(models) == m)
                if sel.sum() >= min_per_model:
                    shifts[m] = eco.predictor.fit_shift(m, X[sel], y[sel])
        local = LocalOutcomes(
            X[rows], [models[i] for i in np.flatnonzero(rows)], y[rows],
            k=k, strength=strength, min_similarity=min_similarity,
        )  # fmt: skip
        return shifts, local

    def predict(rows: np.ndarray, local: LocalOutcomes | None) -> np.ndarray:
        idx = {n: j for j, n in enumerate(eco.predictor.names)}
        P = eco.predictor.predict_proba(X[rows])
        out = []
        for i, row in zip(np.flatnonzero(rows), P):
            p = {n: float(row[j]) for n, j in idx.items()}
            if local is not None:
                p = local.adjust(X[i], p)
            out.append(p[models[i]])
        return np.array(out)

    if test.sum() >= 10 and (~test).sum() >= 10:
        for n, s in base_shifts.items():
            eco.predictor.anchors[n].shift = s
        prior = predict(test, None)
        shifts, local = fit(~test)
        shifted = predict(test, None)
        adapted = predict(test, local)
        yt = y[test]
        report["test"] = {
            "records": int(test.sum()),
            "brier_prior": brier(prior, yt),
            "brier_shifts": brier(shifted, yt),
            "brier_adapted": brier(adapted, yt),
            "success_rate": float(yt.mean()),
        }
        report["test"]["helps"] = report["test"]["brier_adapted"] < report["test"]["brier_prior"]
    else:
        report["test"] = None  # too little data to check; fitted on everything anyway

    shifts, local = fit(np.ones(len(records), bool))
    eco.shifts, eco.local = shifts, local
    report["shifts"] = shifts
    return report
