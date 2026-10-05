"""RouterBench (withmartian/routerbench): ~36K prompts x 11 models, 8 benchmarks.

Raw layout (pickled DataFrame): sample_id, prompt, eval_name, then per model:
  <model>                 correctness (0/1, or a 0..1 score for MT-Bench)
  <model>|model_response  the answer text
  <model>|total_cost      USD cost of that answer
"""

from __future__ import annotations

import pandas as pd

from ecoroute.data.model_names import canonical
from ecoroute.data.schema import OUTCOME_COLUMNS, assign_split, conform, prompt_id

HF_ID = "withmartian/routerbench"
FILES = {"0shot": "routerbench_0shot.pkl", "5shot": "routerbench_5shot.pkl"}


def fetch_raw(variant: str = "0shot") -> pd.DataFrame:
    """Download from Hugging Face (needs network access to huggingface.co).

    The file is a pickle, which can run code when loaded. We only load it from the
    official dataset repo.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_ID, FILES[variant], repo_type="dataset")
    return pd.read_pickle(path)


def model_columns(raw: pd.DataFrame) -> list[str]:
    return [c[: -len("|total_cost")] for c in raw.columns if c.endswith("|total_cost")]


def _prompt_text(value) -> str:
    # A few RouterBench prompts are stored as a list of turns (MT-Bench).
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(v) for v in value)
    return str(value)


def to_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = model_columns(raw)
    for rec in raw.to_dict("records"):
        text = _prompt_text(rec["prompt"])
        pid = prompt_id(text)
        for m in models:
            score = rec.get(m)
            if score is None or pd.isna(score):
                continue
            rows.append(
                {
                    "prompt_id": pid,
                    "source": "routerbench",
                    "source_id": str(rec["sample_id"]),
                    "task": str(rec["eval_name"]),
                    "prompt": text,
                    "model": canonical(m),
                    "correct": float(score),
                    "in_tokens": None,
                    "out_tokens": None,
                    "cost_usd": float(rec[f"{m}|total_cost"]),
                    "split": assign_split(pid),
                }
            )
    return conform(pd.DataFrame(rows, columns=list(OUTCOME_COLUMNS)), OUTCOME_COLUMNS)
