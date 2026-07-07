"""Reproduce the project's headline numbers (seeded, config-driven).

Run:  .venv/bin/python scripts/headline_numbers.py

1. PERFECT-FORESIGHT CEILING (5-regime Monte Carlo panel, n=300 positions):
   the oracle exit's mean-P&L uplift over hold-to-expiry, per cost level.
   This quantifies the paper's "optimal stopping" claim with validated
   accounting -- as a CEILING, not a strategy.

2. CAUSAL CAPTURE WHERE EDGE IS DETECTABLE (regime-shift world, n=40):
   sigma_real 0.32 -> 0.10 at mid-life while marks stay at IV=0.20; the
   causal vol-forecast exit rule's capture of the oracle's edge over hold,
   with entry-date cluster-bootstrap CI, net of transaction costs.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from gamma_exit.analytics.metrics import bootstrap_capture_ci, capture_fraction
from gamma_exit.backtest.runner import run
from gamma_exit.backtest.synthetic import Candidate, SyntheticSource, quotes_from_path
from gamma_exit.config import load_config
from gamma_exit.pnl.engine import simulate_gbm_paths


class _ListSource:
    def __init__(self, candidates):
        self._candidates = candidates

    def positions(self):
        return self._candidates


def five_regime_panel(cfg) -> None:
    src = SyntheticSource(
        n_per_scenario=60,
        r=cfg.rates.risk_free,
        q=cfg.rates.dividend_yield,
        base_seed=cfg.experiment.seed,
    )
    res = run(cfg, src)
    z = res[res["entry_protocol"] == "unconditional"]
    for cl in ("zero", "full_spread"):
        m = z[z["cost_level"] == cl].pivot_table(
            index="position_id", columns="policy", values="pnl"
        )
        o, h = m["oracle"].mean(), m["hold_to_expiry"].mean()
        print(
            f"[5-regime, n={len(m)}] {cl}: oracle {o:+.4f} vs hold {h:+.4f} "
            f"-> uplift {(o - h) / abs(h) * 100:+.1f}%  "
            f"(oracle win rate {(m['oracle'] > 0).mean():.0%})"
        )


def edge_decay_world(cfg) -> None:
    cands = []
    for i in range(40):
        seed = 90_000 + 11 * i
        start = np.busday_offset(
            date(2026, 1, 5) + timedelta(weeks=i % 26), 0, roll="forward"
        ).item()
        a = simulate_gbm_paths(100.0, 0.05, 0.32, 63 / 252, 63, 1, seed)[0]
        b = simulate_gbm_paths(float(a[-1]), 0.05, 0.10, 63 / 252, 63, 1, seed + 1)[0]
        quotes, spec = quotes_from_path(
            np.concatenate([a, b[1:]]), k=100.0, sigma_iv=0.20,
            r=cfg.rates.risk_free, start=start,
        )
        pre_path = simulate_gbm_paths(100.0, 0.05, 0.32, 1.0, 251, 1, seed + 2)[0]
        pre_dates = pd.bdate_range(end=start - timedelta(days=1), periods=252).date
        pre = pd.Series(pre_path * (100.0 / pre_path[-1]), index=pre_dates)
        cands.append(Candidate(quotes, spec, pre, "edge_decay", seed))

    res = run(cfg, _ListSource(cands))
    d = res[res["entry_protocol"] == "unconditional"]
    for cl in ("zero", "full_spread"):
        sub = d[d["cost_level"] == cl]
        m = sub.pivot_table(index="position_id", columns="policy", values="pnl")
        v, h = m["vol_regime_1.00"].mean(), m["hold_to_expiry"].mean()
        cap = capture_fraction(sub, "vol_regime_1.00")
        lo, hi = bootstrap_capture_ci(sub, "vol_regime_1.00", n_boot=2000, seed=1)
        print(
            f"[edge-decay, n={len(m)}] {cl}: vol_regime {v:+.4f} vs hold {h:+.4f} "
            f"-> capture {cap:.2f} of oracle edge, CI90 [{lo:+.2f}, {hi:+.2f}]"
        )


if __name__ == "__main__":
    cfg = load_config()
    five_regime_panel(cfg)
    edge_decay_world(cfg)
