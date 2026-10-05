"""RouterArena (RouteWorks/RouterArena): 8.4K prompts, 9 domains, easy/medium/hard labels.

It has no per-model outcomes, so it feeds the prompts table: our own free evaluation
runs, and a check that predicted difficulty agrees with the labelled difficulty.
"""

from __future__ import annotations

import string

import pandas as pd

from ecoroute.data.schema import PROMPT_COLUMNS, assign_split, conform, prompt_id

HF_ID = "RouteWorks/RouterArena"


def fetch_raw(split: str = "full") -> pd.DataFrame:
    """Download from Hugging Face (needs network access to huggingface.co)."""
    from datasets import load_dataset

    return load_dataset(HF_ID, split=split).to_pandas()


def format_prompt(context: str, question: str, options) -> str:
    parts = [p for p in (context, question) if p and str(p).strip()]
    opts = list(options) if options is not None else []
    if opts:
        letters = string.ascii_uppercase
        parts.append("\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts)))
    return "\n\n".join(str(p) for p in parts)


def to_prompts(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in raw.to_dict("records"):
        text = format_prompt(rec.get("Context"), rec["Question"], rec.get("Options"))
        pid = prompt_id(text)
        rows.append(
            {
                "prompt_id": pid,
                "source": "routerarena",
                "source_id": rec["Global Index"],
                "task": rec["Dataset name"],
                "domain": rec["Domain"],
                "difficulty": rec["Difficulty"],
                "prompt": text,
                "answer": rec["Answer"],
                "split": assign_split(pid),
            }
        )
    return conform(pd.DataFrame(rows, columns=list(PROMPT_COLUMNS)), PROMPT_COLUMNS)
