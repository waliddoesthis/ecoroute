"""SPROUT (CARROT-LLM-Routing/SPROUT): ~44K prompts x 13 models, judge-graded.

Raw layout: one row per prompt; every model is its own column holding a struct
{judge_response, num_input_tokens, num_output_tokens, response, score}.
"""

from __future__ import annotations

import pandas as pd

from ecoroute.data.model_names import canonical
from ecoroute.data.prices import token_cost
from ecoroute.data.schema import OUTCOME_COLUMNS, assign_split, conform, prompt_id

HF_ID = "CARROT-LLM-Routing/SPROUT"
META_COLUMNS = {"key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"}


def fetch_raw(split: str | None = None) -> pd.DataFrame:
    """Download from Hugging Face (needs network access to huggingface.co)."""
    from datasets import load_dataset

    ds = load_dataset(HF_ID)
    splits = [split] if split else list(ds.keys())
    return pd.concat([ds[s].to_pandas() for s in splits], ignore_index=True)


def model_columns(raw: pd.DataFrame) -> list[str]:
    return [c for c in raw.columns if c not in META_COLUMNS]


def to_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in raw.to_dict("records"):
        pid = prompt_id(rec["prompt"])
        for col in model_columns(raw):
            cell = rec[col]
            model = canonical(col)
            if not isinstance(cell, dict) or cell.get("score") is None:
                continue
            rows.append(
                {
                    "prompt_id": pid,
                    "source": "sprout",
                    "source_id": rec["key"],
                    "task": rec["dataset"],
                    "prompt": rec["prompt"],
                    "model": model,
                    "correct": float(cell["score"]),
                    "in_tokens": cell.get("num_input_tokens"),
                    "out_tokens": cell.get("num_output_tokens"),
                    # SPROUT records tokens, not prices: cost uses its published price table.
                    "cost_usd": token_cost(
                        model, cell.get("num_input_tokens"), cell.get("num_output_tokens")
                    ),
                    "split": assign_split(pid),
                }
            )
    return conform(pd.DataFrame(rows, columns=list(OUTCOME_COLUMNS)), OUTCOME_COLUMNS)
