"""Measure the confidentiality check: recall on labelled PII, false alarms on ordinary prompts.

    python scripts/eval_confidentiality.py --n 2000 --data data/processed

Recall comes from the ai4privacy pii-masking-400k validation split (English): a text
whose labels include a restricted type (cards, IDs, passwords, ...) must come out
restricted, and any labelled text must come out at least confidential. False alarms come
from benchmark prompts in data/processed (which should hold no personal data): the share
raised above internal. Reports rules alone and rules + NER, plus time per prompt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from ecoroute.confidentiality import CallerPolicyLayer, Detector, Level, NERLayer, RulesLayer
from ecoroute.confidentiality.ner import DEFAULT_MODEL, LABEL_LEVELS, hf_tagger


def expected_level(labels: list[str]) -> Level:
    levels = [LABEL_LEVELS.get(lab.upper().rstrip("0123456789")) for lab in labels]
    known = [lv for lv in levels if lv is not None]
    return max(known) if known else Level.INTERNAL


def evaluate(det: Detector, texts: list[str], expected: list[Level]) -> dict:
    t0 = time.perf_counter()
    got = [det.classify(t).level for t in texts]
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(texts))
    restricted = [g for g, e in zip(got, expected) if e == Level.RESTRICTED]
    pii = [g for g, e in zip(got, expected) if e >= Level.CONFIDENTIAL]
    return {
        "n": len(texts),
        "restricted_recall": sum(g == Level.RESTRICTED for g in restricted)
        / max(1, len(restricted)),
        "pii_recall": sum(g >= Level.CONFIDENTIAL for g in pii) / max(1, len(pii)),
        "ms_per_prompt": ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    rules = Detector([CallerPolicyLayer(), RulesLayer()])
    full = Detector([CallerPolicyLayer(), RulesLayer(), NERLayer(hf_tagger(args.model))])
    report = {}
    for name, det in (("rules", rules), ("rules+ner", full)):
        r = evaluate(det, texts, expected)
        r["false_alarm_rate"] = sum(det.classify(t).level > Level.INTERNAL for t in benign) / len(
            benign
        )
        report[name] = r
        print(name, json.dumps(r, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
