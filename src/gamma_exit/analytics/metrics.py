"""Result metrics. The headline number of the whole study is CAPTURE:

    capture = (mean policy P&L - mean hold P&L) / (mean oracle P&L - mean hold P&L)

-- the fraction of the non-tradable ceiling's edge over hold-to-expiry that
a CAUSAL rule actually collects, per cost level. Everything else here
(means, Sharpe, win rates, stop-time stats) is supporting detail.

A note on inference: positions whose holding windows overlap share one realized
path, so rows are NOT independent. Confidence intervals therefore come from
a cluster bootstrap that resamples ENTRY DATES (whole clusters of positions)
with replacement -- the effective sample size is the number of distinct
entry dates, not the number of rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLD = "hold_to_expiry"
ORACLE = "oracle"


def _matched(results: pd.DataFrame) -> pd.DataFrame:
    """position_id x policy P&L matrix for ONE cost level / entry protocol,
    keeping only positions that have every policy (matched comparison)."""
    pivot = results.pivot_table(index="position_id", columns="policy", values="pnl")
    return pivot.dropna()


def capture_fraction(results: pd.DataFrame, policy: str) -> float:
    """Fraction of the oracle's edge over hold captured by `policy`.

    nan when the oracle has no edge (denominator <= 0): in a window where
    even perfect foresight cannot beat holding, capture is undefined -- do
    not report a number that pretends otherwise.
    """
    m = _matched(results)
    edge_oracle = m[ORACLE].mean() - m[HOLD].mean()
    if edge_oracle <= 0:
        return float("nan")
    return float((m[policy].mean() - m[HOLD].mean()) / edge_oracle)


def bootstrap_capture_ci(
    results: pd.DataFrame,
    policy: str,
    n_boot: int = 2000,
    level: float = 0.90,
    seed: int = 0,
) -> tuple[float, float]:
    """Entry-date cluster bootstrap CI for the capture fraction."""
    df = results[["position_id", "entry_date", "policy", "pnl"]]
    m = _matched(df).join(
        df.drop_duplicates("position_id").set_index("position_id")["entry_date"]
    )
    clusters = [g[[HOLD, ORACLE, policy]].to_numpy() for _, g in m.groupby("entry_date")]
    rng = np.random.default_rng(seed)
    stats = []
    k = len(clusters)
    for _ in range(n_boot):
        sample = np.concatenate([clusters[i] for i in rng.integers(0, k, k)])
        hold, oracle, pol = sample[:, 0].mean(), sample[:, 1].mean(), sample[:, 2].mean()
        if oracle - hold > 0:
            stats.append((pol - hold) / (oracle - hold))
    if not stats:
        return (float("nan"), float("nan"))
    alpha = (1 - level) / 2
    return (
        float(np.quantile(stats, alpha)),
        float(np.quantile(stats, 1 - alpha)),
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per (cost_level, policy): the paper's Table-1-style summary plus
    capture with a cluster-bootstrap CI. Expects one entry protocol."""
    out = []
    for cost_level, sub in results.groupby("cost_level", sort=False):
        m = _matched(sub)
        for policy in m.columns:
            pnls = m[policy]
            row = {
                "cost_level": cost_level,
                "policy": policy,
                "n": len(pnls),
                "mean_pnl": pnls.mean(),
                "median_pnl": pnls.median(),
                "std_pnl": pnls.std(ddof=1),
                "sharpe": pnls.mean() / pnls.std(ddof=1) if pnls.std(ddof=1) > 0 else np.nan,
                "win_rate": (pnls > 0).mean(),
                "mean_exit_frac": sub[sub["policy"] == policy]["exit_frac"].mean(),
                "vs_hold": pnls.mean() - m[HOLD].mean(),
            }
            if policy != HOLD:
                row["capture"] = capture_fraction(sub, policy)
                lo, hi = bootstrap_capture_ci(sub, policy, n_boot=500)
                row["capture_ci90"] = f"[{lo:+.2f}, {hi:+.2f}]"
            out.append(row)
    return pd.DataFrame(out).set_index(["cost_level", "policy"]).sort_index()
