"""BigCodeBench (bigcode/bigcodebench-perf): 1,140 Python tasks x ~130 models, pass@1.

Our main source (SPROUT) has no coding prompts, so the router could not tell a coding
request from chat. BigCodeBench publishes a 0/1 matrix (one row per model, one column per
task) that covers 10 of our 13 SPROUT models; the task text comes from bigcode/bigcodebench
(apache-2.0), joined on task_id. The hard subset (148 tasks) adds llama-3.1-405b.

We use the "instruct" split: its prompts are natural-language requests, like what users
send, while "complete" gives a function signature and docstring to finish.
"""

from __future__ import annotations

import pandas as pd

from ecoroute.data.prices import token_cost
from ecoroute.data.schema import OUTCOME_COLUMNS, assign_split, conform, prompt_id

PERF_IDS = ("bigcode/bigcodebench-perf", "bigcode/bigcodebench-hard-perf")
PROMPTS_ID = "bigcode/bigcodebench"
PROMPTS_SPLIT = "v0.1.4"

# BigCodeBench model name -> our canonical name. Where a model has several dated versions,
# the first listed that is present wins (the version closest to the one SPROUT graded).
MODEL_MAP: dict[str, list[str]] = {
    "gpt-4o": ["GPT-4o-2024-05-13", "GPT-4o-2024-08-06", "GPT-4o-2024-11-20"],
    "gpt-4o-mini": ["GPT-4o-mini-2024-07-18"],
    "claude-3.5-sonnet": ["Claude-3.5-Sonnet-20240620", "Claude-3.5-Sonnet-20241022"],
    "llama-3.1-8b-instruct": ["Llama-3.1-8B-Instruct"],
    "llama-3.1-70b-instruct": ["Llama-3.1-70B-Instruct"],
    "llama-3.1-405b-instruct": ["Llama-3.1-405B-Instruct"],
    "llama-3.3-70b-instruct": ["Llama-3.3-70B-Instruct"],
    "llama-3.2-1b-instruct": ["Llama-3.2-1B-Instruct"],
    "llama-3.2-3b-instruct": ["Llama-3.2-3B-Instruct"],
    "granite-3-8b-instruct": ["Granite-3.0-8B-Instruct"],
    "granite-3-2b-instruct": ["Granite-3.0-2B-Instruct"],
}

# The perf tables record pass/fail only. Costs are estimated from the prompt length and a
# typical answer length so code prompts can be ranked by price like the rest.
ASSUMED_OUT_TOKENS = 400


def fetch_raw(split: str = "instruct") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(perf matrix with a 'Model' column per row, tasks with task_id and prompts).

    Reads the parquet files directly: the datasets viewer fails on 1,141 columns.
    Needs network access to huggingface.co.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    perf = []
    for repo in PERF_IDS:
        path = hf_hub_download(repo, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        perf.append(pd.read_parquet(path))
    tasks = load_dataset(PROMPTS_ID, split=PROMPTS_SPLIT).to_pandas()
    return _merge_perf(perf), tasks


def _merge_perf(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per model across the full and hard tables (full results win on overlap)."""
    long = []
    for t in tables:
        m = t.melt(id_vars="Model", var_name="task_id", value_name="passed")
        long.append(m.dropna(subset=["passed"]))
    out = pd.concat(long, ignore_index=True).drop_duplicates(["Model", "task_id"], keep="first")
    return out.pivot(index="Model", columns="task_id", values="passed").reset_index()


def pick_models(perf: pd.DataFrame) -> dict[str, str]:
    """canonical name -> the BigCodeBench row used for it."""
    present = set(perf.Model)
    chosen = {}
    for name, candidates in MODEL_MAP.items():
        hit = next((c for c in candidates if c in present), None)
        if hit is not None:
            chosen[name] = hit
    return chosen


def to_outcomes(perf: pd.DataFrame, tasks: pd.DataFrame, prompt_col: str = "instruct_prompt"):
    prompts = dict(zip(tasks.task_id, tasks[prompt_col]))
    rows = []
    for model, row_name in pick_models(perf).items():
        rec = perf.loc[perf.Model == row_name].iloc[0]
        for task_id, passed in rec.drop("Model").items():
            text = prompts.get(task_id)
            if text is None or pd.isna(passed):
                continue
            pid = prompt_id(text)
            in_tokens = max(1, len(text) // 4)
            rows.append(
                {
                    "prompt_id": pid,
                    "source": "bigcodebench",
                    "source_id": task_id,
                    "task": "bigcodebench/instruct",
                    "prompt": text,
                    "model": model,
                    "correct": float(passed),
                    "in_tokens": in_tokens,
                    "out_tokens": ASSUMED_OUT_TOKENS,
                    "cost_usd": token_cost(model, in_tokens, ASSUMED_OUT_TOKENS),
                    "split": assign_split(pid),
                }
            )
    return conform(pd.DataFrame(rows, columns=list(OUTCOME_COLUMNS)), OUTCOME_COLUMNS)
