"""How good are the predictions, and what do they buy when used for routing?

Quality: Brier score, ECE (calibration) and mean per-model AUC, on observed cells.
Routing: route each prompt to the cheapest model whose predicted P(correct) >= tau,
sweep tau, and compare accuracy and cost against single models and the oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def brier(P: np.ndarray, Y: np.ndarray) -> float:
    seen = ~np.isnan(Y)
    return float(np.mean((P[seen] - Y[seen]) ** 2))


def ece(P: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: gap between predicted and observed accuracy per bin."""
    seen = ~np.isnan(Y)
    p, y = P[seen], Y[seen]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def mean_auc(P: np.ndarray, Y: np.ndarray) -> float:
    """Average over models of the AUC for "answered correctly" (score >= 0.5)."""
    aucs = []
    for j in range(Y.shape[1]):
        seen = ~np.isnan(Y[:, j])
        labels = Y[seen, j] >= 0.5
        if labels.any() and (~labels).any():
            aucs.append(roc_auc_score(labels, P[seen, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def quality_report(P: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    return {"brier": brier(P, Y), "ece": ece(P, Y), "auc": mean_auc(P, Y)}


def route_cheapest_above(P: np.ndarray, C: np.ndarray, tau: float) -> np.ndarray:
    """Index of the cheapest model with P >= tau per prompt; the most likely model if none."""
    ok = P >= tau
    masked_cost = np.where(ok, C, np.inf)
    choice = masked_cost.argmin(axis=1)
    none_ok = ~ok.any(axis=1)
    choice[none_ok] = P[none_ok].argmax(axis=1)
    return choice


def routing_curve(
    P: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    models: list[str],
    taus: np.ndarray | None = None,
) -> pd.DataFrame:
    """Accuracy and mean cost per prompt for the router at each tau, plus references.

    Only prompts where every model has both an outcome and a cost are used, so all
    rows compare the same prompts.
    """
    full = ~np.isnan(Y).any(axis=1) & ~np.isnan(C).any(axis=1)
    P, Y, C = P[full], Y[full], C[full]
    rows_idx = np.arange(len(Y))
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)
    out = []
    for tau in taus:
        pick = route_cheapest_above(P, C, tau)
        out.append(
            {
                "policy": f"router tau={tau:.2f}",
                "accuracy": Y[rows_idx, pick].mean(),
                "cost": C[rows_idx, pick].mean(),
            }
        )
    for j, m in enumerate(models):
        out.append({"policy": f"always {m}", "accuracy": Y[:, j].mean(), "cost": C[:, j].mean()})
    # Oracle: cheapest model that got it right; cheapest model overall when none did.
    correct = Y >= 0.5
    oracle = np.where(
        correct.any(axis=1), np.where(correct, C, np.inf).argmin(axis=1), C.argmin(axis=1)
    )
    out.append(
        {
            "policy": "oracle",
            "accuracy": Y[rows_idx, oracle].mean(),
            "cost": C[rows_idx, oracle].mean(),
        }
    )
    df = pd.DataFrame(out)
    df.attrs["n_prompts"] = int(len(Y))
    return df


def savings_at_quality(
    curve: pd.DataFrame, reference: str, tolerance: float = 0.01
) -> dict[str, float | str | bool]:
    """Cheapest router point within `tolerance` accuracy of the reference model.

    tolerance=0.01 means "at most 1 point less accurate than always using the reference".
    """
    ref = curve.loc[curve.policy == reference].iloc[0]
    router = curve[
        curve.policy.str.startswith("router") & (curve.accuracy >= ref.accuracy - tolerance)
    ]
    if router.empty:
        return {"reference": reference, "tolerance": tolerance, "matched": False}
    best = router.loc[router.cost.idxmin()]
    return {
        "reference": reference,
        "tolerance": tolerance,
        "matched": True,
        "router_policy": best.policy,
        "router_accuracy": float(best.accuracy),
        "reference_accuracy": float(ref.accuracy),
        "cost_saving_pct": float(100 * (1 - best.cost / ref.cost)) if ref.cost > 0 else 0.0,
    }


def difficulty_buckets(Y: np.ndarray) -> np.ndarray:
    """Label each prompt easy / medium / hard by the share of models that answered it.

    hard: at most 25% of models correct; easy: at least 75%; medium otherwise. This is
    known only after the fact, so it is for evaluation, never for routing.
    """
    share = np.nanmean(Y >= 0.5, axis=1)
    return np.where(share <= 0.25, "hard", np.where(share >= 0.75, "easy", "medium"))


def breakdown_by_difficulty(
    P: np.ndarray, Y: np.ndarray, C: np.ndarray, tau: float
) -> pd.DataFrame:
    """Router (at tau) vs. oracle accuracy and cost per difficulty bucket.

    The gap column shows where the router loses accuracy; the 2026 Routing Plateau study
    found most of the gap to the oracle sits in the hard bucket.
    """
    full = ~np.isnan(Y).any(axis=1) & ~np.isnan(C).any(axis=1)
    P, Y, C = P[full], Y[full], C[full]
    idx = np.arange(len(Y))
    pick = route_cheapest_above(P, C, tau)
    correct = Y >= 0.5
    oracle = np.where(
        correct.any(axis=1), np.where(correct, C, np.inf).argmin(axis=1), C.argmin(axis=1)
    )
    buckets = difficulty_buckets(Y)
    rows = []
    for b in ("easy", "medium", "hard"):
        sel = buckets == b
        if not sel.any():
            continue
        rows.append(
            {
                "bucket": b,
                "share": float(sel.mean()),
                "router_accuracy": float(Y[idx[sel], pick[sel]].mean()),
                "oracle_accuracy": float(Y[idx[sel], oracle[sel]].mean()),
                "router_cost": float(C[idx[sel], pick[sel]].mean()),
                "oracle_cost": float(C[idx[sel], oracle[sel]].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out.oracle_accuracy - out.router_accuracy
    return out


def evaluate_at_tau(P: np.ndarray, Y: np.ndarray, C: np.ndarray, tau: float) -> dict[str, float]:
    """Accuracy and mean cost of the router at a fixed tau, on prompts with full rows."""
    full = ~np.isnan(Y).any(axis=1) & ~np.isnan(C).any(axis=1)
    P, Y, C = P[full], Y[full], C[full]
    pick = route_cheapest_above(P, C, tau)
    idx = np.arange(len(Y))
    return {"accuracy": float(Y[idx, pick].mean()), "cost": float(C[idx, pick].mean())}


def held_out_saving(
    P_val: np.ndarray,
    Y_val: np.ndarray,
    C_val: np.ndarray,
    P_test: np.ndarray,
    Y_test: np.ndarray,
    C_test: np.ndarray,
    models: list[str],
    tolerance: float = 0.01,
) -> dict[str, float | str | bool]:
    """Pick tau on validation, then measure it on test.

    Choosing tau on the test set (as savings_at_quality does) is optimistic, because the
    best of 19 thresholds is picked after seeing the answers. This is the honest number.
    """
    val_curve = routing_curve(P_val, Y_val, C_val, models)
    singles = val_curve[val_curve.policy.str.startswith("always")]
    reference = singles.loc[singles.accuracy.idxmax()].policy
    chosen = savings_at_quality(val_curve, reference, tolerance)
    if not chosen["matched"]:
        return {"reference": reference, "matched": False}
    tau = float(chosen["router_policy"].split("=")[1])
    test_curve = routing_curve(P_test, Y_test, C_test, models)
    ref = test_curve.loc[test_curve.policy == reference].iloc[0]
    got = evaluate_at_tau(P_test, Y_test, C_test, tau)
    return {
        "reference": reference,
        "matched": True,
        "tau": tau,
        "router_accuracy": got["accuracy"],
        "reference_accuracy": float(ref.accuracy),
        "cost_saving_pct": float(100 * (1 - got["cost"] / ref.cost)) if ref.cost > 0 else 0.0,
    }
