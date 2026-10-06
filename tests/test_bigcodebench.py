import pandas as pd

from ecoroute.data import bigcodebench


def test_bigcodebench_outcomes_from_perf_matrix():
    full = pd.DataFrame(
        {"Model": ["GPT-4o-2024-05-13", "GPT-4o-2024-11-20", "Llama-3.1-8B-Instruct", "Other"],
         "BigCodeBench/0": [1, 0, 0, 1], "BigCodeBench/1": [1, 1, 0, 0]}
    )  # fmt: skip
    hard = pd.DataFrame({"Model": ["Llama-3.1-405B-Instruct"], "BigCodeBench/1": [1]})
    perf = bigcodebench._merge_perf([full, hard])
    tasks = pd.DataFrame(
        {"task_id": ["BigCodeBench/0", "BigCodeBench/1"],
         "instruct_prompt": ["Write a function that sorts a list.", "Parse a CSV file."]}
    )  # fmt: skip
    out = bigcodebench.to_outcomes(perf, tasks)
    # one gpt-4o version (the first listed), llama 8b, and 405b from the hard table only
    assert set(out.model) == {"gpt-4o", "llama-3.1-8b-instruct", "llama-3.1-405b-instruct"}
    gpt = out[out.model == "gpt-4o"].set_index("source_id").correct
    assert gpt.to_dict() == {"BigCodeBench/0": 1.0, "BigCodeBench/1": 1.0}
    assert len(out[out.model == "llama-3.1-405b-instruct"]) == 1  # NaN task skipped
    assert (out.source == "bigcodebench").all() and out.cost_usd.notna().all()
    assert out.groupby("prompt_id").split.nunique().max() == 1
