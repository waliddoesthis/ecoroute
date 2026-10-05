"""Per-token prices for models in the public datasets, used where a source records
token counts but not cost (SPROUT).

SPROUT prices are the ones its authors published with the dataset (CARROT paper,
arXiv 2502.03261, Table 2), so our cost numbers stay comparable with theirs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# canonical model name -> (USD per 1M input tokens, USD per 1M output tokens)
SOURCE_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-3.5-sonnet": (3.0, 15.0),
    "titan-text-premier": (0.5, 1.5),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "granite-3-2b-instruct": (0.1, 0.1),
    "granite-3-8b-instruct": (0.2, 0.2),
    "llama-3.1-70b-instruct": (0.9, 0.9),
    "llama-3.1-8b-instruct": (0.2, 0.2),
    "llama-3.2-1b-instruct": (0.06, 0.06),
    "llama-3.2-3b-instruct": (0.06, 0.06),
    "llama-3.3-70b-instruct": (0.9, 0.9),
    "mixtral-8x7b-instruct": (0.6, 0.6),
    "llama-3.1-405b-instruct": (3.5, 3.5),
}


def token_cost(model: str, in_tokens, out_tokens) -> float:
    """USD cost of one answer, NaN when the model has no price or tokens are missing."""
    price = SOURCE_PRICES_PER_MTOK.get(model)
    if price is None or pd.isna(in_tokens) or pd.isna(out_tokens):
        return float("nan")
    return (float(in_tokens) * price[0] + float(out_tokens) * price[1]) / 1e6


def fill_missing_costs(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Estimate cost_usd from token counts wherever it is missing. Observed costs are kept."""
    missing = outcomes.cost_usd.isna()
    if not missing.any():
        return outcomes
    out = outcomes.copy()
    sub = out.loc[missing]
    p_in = sub.model.map(lambda m: SOURCE_PRICES_PER_MTOK.get(m, (np.nan, np.nan))[0])
    p_out = sub.model.map(lambda m: SOURCE_PRICES_PER_MTOK.get(m, (np.nan, np.nan))[1])
    est = (sub.in_tokens.astype("float64") * p_in + sub.out_tokens.astype("float64") * p_out) / 1e6
    out.loc[missing, "cost_usd"] = est.to_numpy()
    return out
