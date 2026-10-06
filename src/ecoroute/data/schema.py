"""Unified tables every loader produces.

outcomes: one row per (prompt, model) pair with a graded result. This is what the
          quality predictors train on.
prompts:  one row per prompt with no model outcomes (e.g. RouterArena). Used for our
          own free evaluation runs and for checking predicted difficulty.
"""

from __future__ import annotations

import hashlib
import unicodedata

import pandas as pd

OUTCOME_COLUMNS: dict[str, str] = {
    "prompt_id": "string",  # content hash of the normalised prompt, shared across sources
    "source": "string",  # sprout | routerbench | ...
    "source_id": "string",  # the row's id inside its source
    "task": "string",  # benchmark / eval the prompt comes from
    "prompt": "string",
    "model": "string",  # canonical model name, see model_names.py
    "correct": "float64",  # 0..1
    "in_tokens": "Int64",  # nullable: not every source records it
    "out_tokens": "Int64",
    "cost_usd": "float64",  # observed cost of this answer, NaN when unknown
    "split": "string",  # train | val | test
}

PROMPT_COLUMNS: dict[str, str] = {
    "prompt_id": "string",
    "source": "string",
    "source_id": "string",
    "task": "string",
    "domain": "string",
    "difficulty": "string",  # labelled difficulty when the source has one
    "prompt": "string",
    "answer": "string",
    "split": "string",
}

SPLITS = ("train", "val", "test")


def normalise_prompt(text: str) -> str:
    """Normalise so the same prompt from two sources hashes to the same id."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).lower()


def prompt_id(text: str) -> str:
    return hashlib.sha256(normalise_prompt(text).encode("utf-8")).hexdigest()[:16]


def assign_split(pid: str, val: float = 0.1, test: float = 0.1) -> str:
    """Deterministic split from the prompt id.

    Splitting on the prompt (not the row) keeps every model's answer to one prompt in
    the same split, and a prompt found in two sources can't leak from train into test.
    """
    bucket = int(pid[:8], 16) / 0xFFFFFFFF
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


def dedupe_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated (prompt, model) rows inside one source into one row.

    Sources contain the same prompt text under several ids (SPROUT has ~3.6k such
    pairs, sometimes graded differently). Left in, they double-count those prompts and
    give a model two labels for one input. Grades and costs are averaged; the first
    row's metadata is kept.
    """
    keys = ["source", "prompt_id", "model"]
    if not df.duplicated(keys).any():
        return df
    numeric = {"correct": "mean", "in_tokens": "mean", "out_tokens": "mean", "cost_usd": "mean"}
    agg = {c: numeric.get(c, "first") for c in df.columns if c not in keys}
    out = df.groupby(keys, sort=False, as_index=False, dropna=False).agg(agg)
    for c in ("in_tokens", "out_tokens"):
        out[c] = out[c].round()
    return out[list(df.columns)]


def conform(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Order, type and validate a frame against a schema."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[list(columns)].astype(columns)
    if columns is OUTCOME_COLUMNS:
        out = dedupe_outcomes(out).astype(columns)
    if "correct" in out and not out["correct"].dropna().between(0, 1).all():
        raise ValueError("correct must be within [0, 1]")
    if "split" in out and not out["split"].isin(SPLITS).all():
        raise ValueError(f"split must be one of {SPLITS}")
    return out.reset_index(drop=True)
