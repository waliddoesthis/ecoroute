"""Measure the confidentiality check and choose its PII-model settings.

    python scripts/eval_confidentiality.py --n 2000 --data data/processed

Recall comes from the ai4privacy pii-masking-400k validation split (English): a text
whose labels include a restricted type (cards, IDs, passwords, ...) must come out
restricted, and any labelled text must come out at least confidential. False alarms come
from benchmark prompts in data/processed (which should hold no personal data): the share
raised above internal.

Each text is scanned once; the settings are then swept offline:
- min_score: the PII model's confidence needed for a finding to keep its level (findings
  up to 0.2 below it still count, at confidential, as "uncertain means higher");
- weak kinds: kinds that only count next to a second kind of personal data;
- kind_min: per-kind score floors (NERLayer(kind_min=...)) for the noisiest kinds.
The recommended setting is the one with the fewest false alarms that keeps restricted
recall within 1 point of the best and PII recall at or above --min-pii-recall.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecoroute.confidentiality import Level, NERLayer, RulesLayer
from ecoroute.confidentiality.detector import combined_level
from ecoroute.confidentiality.ner import DEFAULT_MODEL, LABEL_LEVELS, WEAK_KINDS, hf_tagger
from ecoroute.confidentiality.rules import Finding

WEAK_SETS = {
    "none": frozenset(),
    "names+cities": frozenset({"givenname", "city"}),
    "all weak kinds": WEAK_KINDS,
}
MIN_SCORES = (0.5, 0.7, 0.85, 0.95)
# Per-kind floors swept on top of min_score 0.5: the weak kinds and passwords cause most
# false alarms on ordinary prompts (run 16: names, cities, building numbers, passwords).
KIND_FLOORS = {
    "weak 0.7": dict.fromkeys(WEAK_KINDS, 0.7),
    "weak 0.85": dict.fromkeys(WEAK_KINDS, 0.85),
    "weak 0.95": dict.fromkeys(WEAK_KINDS, 0.95),
    "weak 0.85, password 0.9": {**dict.fromkeys(WEAK_KINDS, 0.85), "password": 0.9},
    "weak 0.95, password 0.95": {**dict.fromkeys(WEAK_KINDS, 0.95), "password": 0.95},
}


def expected_level(labels: list[str]) -> Level:
    levels = [LABEL_LEVELS.get(lab.upper().rstrip("0123456789")) for lab in labels]
    known = [lv for lv in levels if lv is not None]
    return max(known) if known else Level.INTERNAL


def apply_min_score(
    findings: list[Finding], min_score: float, kind_min: dict[str, float] | None = None
) -> list[Finding]:
    kind_min = kind_min or {}
    out = []
    for f in findings:
        if f.layer == "pii_ner" and f.score < kind_min.get(f.kind, 0.0):
            continue
        if f.layer != "pii_ner" or f.score >= min_score:
            out.append(f)
        elif f.score >= min_score - 0.2:
            out.append(Finding(f.kind, Level.CONFIDENTIAL, f.start, f.end, f.layer, f.score))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-pii-recall", type=float, default=0.95)
    parser.add_argument("--device", type=int, default=None, help="GPU index for the PII model")
    parser.add_argument("--fp16", action="store_true", help="half-precision PII model (GPU)")
    parser.add_argument("--out", type=Path, default=Path("reports/confidentiality.json"))
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("ai4privacy/pii-masking-400k", split="validation")
    ds = ds.filter(lambda r: r["language"] == "en").shuffle(seed=0).select(range(args.n))
    texts = ds["source_text"]
    expected = [expected_level([m["label"] for m in r]) for r in ds["privacy_mask"]]
    benign = (
        pd.read_parquet(args.data / "outcomes.parquet", columns=["prompt_id", "prompt"])
        .drop_duplicates("prompt_id")
        .sample(args.n, random_state=0)
        .prompt.tolist()
    )

    torch.set_grad_enabled(False)
    rules = RulesLayer()
    tagger = hf_tagger(args.model, device=args.device, fp16=args.fp16)
    ner = NERLayer(tagger, threshold=0.0, grey=0.0)  # keep every score
    # Latency as the gateway sees it: one request at a time, timed per prompt.
    ner.scan("warm up the model")
    latencies = []
    for t in benign[: min(300, len(benign))]:
        t0 = time.perf_counter()
        rules.scan(t) + ner.scan(t)
        latencies.append(1000 * (time.perf_counter() - t0))
    p50, p95 = np.percentile(latencies, [50, 95])
    # Throughput when prompts arrive together and are scanned as one batch.
    t0 = time.perf_counter()
    scanned_pii = [r + n for r, n in zip(map(rules.scan, texts), ner.scan_many(texts))]
    scanned_benign = [r + n for r, n in zip(map(rules.scan, benign), ner.scan_many(benign))]
    batched = 1000 * (time.perf_counter() - t0) / (len(texts) + len(benign))
    lengths = sorted(len(t) for t in benign)
    where = f"device={'cpu' if args.device is None else args.device}, fp16={args.fp16}"
    print(
        f"scan time ({where}): one at a time p50 {p50:.1f} ms, p95 {p95:.1f} ms; "
        f"batched {batched:.1f} ms per prompt; "
        f"benchmark prompt length median {lengths[len(lengths) // 2]} chars"
    )
    ms = float(p50)

    def score(settings, use_ner=True):
        min_score, weak, kind_min = settings

        def level(f):
            kept = (
                apply_min_score(f, min_score, kind_min)
                if use_ner
                else [x for x in f if x.layer != "pii_ner"]
            )
            return combined_level(kept, Level.INTERNAL, weak)

        got = [level(f) for f in scanned_pii]
        restricted = [g for g, e in zip(got, expected) if e == Level.RESTRICTED]
        pii = [g for g, e in zip(got, expected) if e >= Level.CONFIDENTIAL]
        alarms, kinds = 0, Counter()
        for f in scanned_benign:
            if level(f) > Level.INTERNAL:
                alarms += 1
                kept = apply_min_score(f, min_score, kind_min) if use_ner else f
                kinds.update({x.kind for x in kept if x.level > Level.INTERNAL})
        return {
            "restricted_recall": sum(g == Level.RESTRICTED for g in restricted)
            / max(1, len(restricted)),
            "pii_recall": sum(g >= Level.CONFIDENTIAL for g in pii) / max(1, len(pii)),
            "false_alarm_rate": alarms / len(benign),
            "false_alarms_by_kind": dict(kinds.most_common(6)),
        }

    rows = [
        {"min_score": None, "weak": "rules only", "kind_min": "-",
         **score((1.0, frozenset(), None), use_ner=False)}
    ]  # fmt: skip
    for min_score in MIN_SCORES:
        for name, weak in WEAK_SETS.items():
            rows.append(
                {"min_score": min_score, "weak": name, "kind_min": "-",
                 **score((min_score, weak, None))}
            )  # fmt: skip
    for name, floors in KIND_FLOORS.items():
        rows.append(
            {"min_score": 0.5, "weak": "none", "kind_min": name,
             **score((0.5, frozenset(), floors))}
        )  # fmt: skip
    table = pd.DataFrame(rows)
    print(
        table.drop(columns="false_alarms_by_kind").to_string(
            index=False, float_format="{:.3f}".format
        )
    )

    with_ner = table[table.min_score.notna()]
    ok = with_ner[
        (with_ner.restricted_recall >= with_ner.restricted_recall.max() - 0.01)
        & (with_ner.pii_recall >= args.min_pii_recall)
    ]
    best = ok.sort_values("false_alarm_rate").iloc[0].to_dict() if len(ok) else None
    print("\nrecommended:", json.dumps(best, indent=2, default=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "ms_per_prompt": ms,
                "p95_ms": float(p95),
                "batched_ms": batched,
                "rows": rows,
                "recommended": best,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
