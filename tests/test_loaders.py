"""Offline tests: each converter gets a tiny frame shaped like its raw source."""

import math

import pandas as pd
import pytest

from ecoroute.data import routerarena, routerbench, sprout
from ecoroute.data.prices import fill_missing_costs
from ecoroute.data.schema import OUTCOME_COLUMNS, PROMPT_COLUMNS, assign_split, prompt_id


def _sprout_cell(score, tin=10, tout=20):
    return {
        "judge_response": "ok",
        "num_input_tokens": tin,
        "num_output_tokens": tout,
        "response": "answer",
        "score": score,
    }


@pytest.fixture
def sprout_raw():
    return pd.DataFrame(
        [
            {
                "key": "k1",
                "dataset": "gsm8k",
                "dataset_level": "",
                "dataset_idx": 0,
                "prompt": "What is 2+2?",
                "golden_answer": "4",
                "openai-gpt-4o": _sprout_cell(1.0),
                "wxai-llama-3-2-1b-instruct": _sprout_cell(0.0),
            },
            {
                "key": "k2",
                "dataset": "mmlu",
                "dataset_level": "",
                "dataset_idx": 1,
                "prompt": "Capital of France?",
                "golden_answer": "Paris",
                "openai-gpt-4o": _sprout_cell(1.0),
                "wxai-llama-3-2-1b-instruct": None,  # missing answer is skipped
            },
        ]
    )


def test_sprout_to_outcomes(sprout_raw):
    out = sprout.to_outcomes(sprout_raw)
    assert list(out.columns) == list(OUTCOME_COLUMNS)
    assert len(out) == 3
    assert set(out.model) == {"gpt-4o", "llama-3.2-1b-instruct"}
    row = out[(out.source_id == "k1") & (out.model == "llama-3.2-1b-instruct")].iloc[0]
    assert row.correct == 0.0 and row.in_tokens == 10 and row.out_tokens == 20
    # llama-3.2-1b: $0.06 / 1M tokens in and out -> 30 tokens cost 1.8e-6
    assert row.cost_usd == pytest.approx(30 * 0.06 / 1e6)


@pytest.fixture
def routerbench_raw():
    return pd.DataFrame(
        [
            {
                "sample_id": "gsm8k.1",
                "prompt": "What is 2+2?",
                "eval_name": "gsm8k",
                "gpt-4-1106-preview": 1.0,
                "gpt-4-1106-preview|model_response": "4",
                "gpt-4-1106-preview|total_cost": 0.002,
                "mistralai/mixtral-8x7b-chat": 0.0,
                "mistralai/mixtral-8x7b-chat|model_response": "5",
                "mistralai/mixtral-8x7b-chat|total_cost": 0.0001,
                "oracle_model_to_route_to": "gpt-4-1106-preview",
            },
            {
                "sample_id": "mtbench.1",
                "prompt": ["Turn one", "Turn two"],
                "eval_name": "mt-bench",
                "gpt-4-1106-preview": 0.9,
                "gpt-4-1106-preview|model_response": "...",
                "gpt-4-1106-preview|total_cost": 0.01,
                "mistralai/mixtral-8x7b-chat": float("nan"),
                "mistralai/mixtral-8x7b-chat|model_response": None,
                "mistralai/mixtral-8x7b-chat|total_cost": 0.0,
                "oracle_model_to_route_to": "gpt-4-1106-preview",
            },
        ]
    )


def test_routerbench_to_outcomes(routerbench_raw):
    out = routerbench.to_outcomes(routerbench_raw)
    assert list(out.columns) == list(OUTCOME_COLUMNS)
    assert len(out) == 3  # NaN correctness skipped
    assert set(out.model) == {"gpt-4-1106-preview", "mixtral-8x7b-instruct"}
    assert out.loc[out.source_id == "mtbench.1", "prompt"].iloc[0] == "Turn one\n\nTurn two"
    assert out.in_tokens.isna().all()


def test_same_prompt_shares_id_across_sources(sprout_raw, routerbench_raw):
    a = sprout.to_outcomes(sprout_raw)
    b = routerbench.to_outcomes(routerbench_raw)
    pid_a = a.loc[a.source_id == "k1", "prompt_id"].iloc[0]
    pid_b = b.loc[b.source_id == "gsm8k.1", "prompt_id"].iloc[0]
    assert pid_a == pid_b
    # mixtral appears in both sources under one canonical name
    assert "mixtral-8x7b-instruct" in set(b.model)


def test_routerarena_to_prompts():
    raw = pd.DataFrame(
        [
            {
                "Category": "02 Library",
                "Domain": "0 Computer science",
                "Dataset name": "ArcMMLU",
                "Global Index": "ArcMMLU_98",
                "Context": "",
                "Question": "Which one?",
                "Options": ["Three", "Four"],
                "Answer": "B",
                "Metadata": "{}",
                "Keywords": "",
                "Difficulty": "easy",
            }
        ]
    )
    out = routerarena.to_prompts(raw)
    assert list(out.columns) == list(PROMPT_COLUMNS)
    assert out.prompt.iloc[0] == "Which one?\n\nA. Three\nB. Four"
    assert out.difficulty.iloc[0] == "easy"


def test_prompt_id_ignores_whitespace_and_case():
    assert prompt_id("What is  2+2?\n") == prompt_id("what is 2+2?")


def test_split_is_deterministic_and_roughly_80_10_10():
    ids = [prompt_id(f"prompt {i}") for i in range(20000)]
    splits = pd.Series([assign_split(i) for i in ids]).value_counts(normalize=True)
    assert abs(splits["train"] - 0.8) < 0.02
    assert abs(splits["val"] - 0.1) < 0.02
    assert abs(splits["test"] - 0.1) < 0.02
    assert assign_split(ids[0]) == assign_split(ids[0])


def test_duplicate_prompts_collapse_to_one_row(sprout_raw):
    dup = sprout_raw.iloc[[0]].assign(key="k3")
    dup.loc[:, "wxai-llama-3-2-1b-instruct"] = [_sprout_cell(1.0, tin=11, tout=21)]
    out = sprout.to_outcomes(pd.concat([sprout_raw, dup], ignore_index=True))
    assert not out.duplicated(["prompt_id", "model"]).any()
    small = out[
        (out.prompt_id == prompt_id("What is 2+2?")) & (out.model == "llama-3.2-1b-instruct")
    ]
    assert small.correct.item() == pytest.approx(0.5)
    assert str(out.in_tokens.dtype) == "Int64"


def test_fill_missing_costs_keeps_observed_and_estimates_rest():
    df = pd.DataFrame(
        {
            "model": ["gpt-4o", "gpt-4o", "unknown-model"],
            "in_tokens": pd.array([1000, 1000, 5], dtype="Int64"),
            "out_tokens": pd.array([500, 500, 5], dtype="Int64"),
            "cost_usd": [0.5, float("nan"), float("nan")],
        }
    )
    out = fill_missing_costs(df)
    assert out.cost_usd.iloc[0] == 0.5
    assert out.cost_usd.iloc[1] == pytest.approx((1000 * 2.5 + 500 * 10.0) / 1e6)
    assert math.isnan(out.cost_usd.iloc[2])
