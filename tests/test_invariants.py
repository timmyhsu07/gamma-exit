"""System-level invariants -- the probes that would catch a wrong pipeline
even when every unit test is green.

1. MARTINGALE WORLD (mu=r, sigma_real=sigma_iv): no policy can have expected
   edge; every tradable rule's mean P&L must be ~0, while the ORACLE's mean
   must be strictly positive -- it argmaxes over noise, which is exactly why
   it is quarantined. If a causal policy showed significant P&L here, the
   pipeline would be leaking the future.

2. DECAYING-EDGE WORLD (vol edge dies mid-life): built so that a causal rule
   watching forecast vol SHOULD beat hold-to-expiry. If vol_regime cannot
   earn positive capture in a world designed for it, the state/forecast
   wiring is broken. (The constant-gap worlds show the opposite -- causal
   exits lose there -- so this pair pins the pipeline from both sides.)

3. DETERMINISM: same seed, same config => byte-identical results frame.

4. DEGENERATE-PARAMETER identities: FixedTime(1.0) == hold_to_expiry exactly;
   puts flow end-to-end through the runner.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_exit.analytics.metrics import capture_fraction
from gamma_exit.backtest.runner import run, run_candidate
from gamma_exit.backtest.synthetic import (
    Candidate,
    Scenario,
    SyntheticSource,
    quotes_from_path,
)
from gamma_exit.config import load_config
from gamma_exit.pnl.engine import simulate_gbm_paths
from gamma_exit.strategy.benchmarks import FixedTime, HoldToExpiry

R = 0.02


class _ListSource:
    """Minimal position source for hand-built worlds."""

    def __init__(self, candidates):
        self._candidates = candidates

    def positions(self):
        return self._candidates


def _pre_history(sigma: float, end_at: float, start: date, seed: int) -> pd.Series:
    n = 252
    path = simulate_gbm_paths(100.0, 0.05, sigma, n / 252, n - 1, 1, seed)[0]
    dates = pd.bdate_range(end=start - timedelta(days=1), periods=n).date
    return pd.Series(path * (end_at / path[-1]), index=dates)


@pytest.fixture(scope="module")
def martingale_results():
    cfg = load_config()
    source = SyntheticSource(
        scenarios=(Scenario("fair", mu=R, sigma_real=0.22, sigma_iv=0.22, n_days=42),),
        n_per_scenario=60,
        r=R,
        base_seed=7100,
    )
    res = run(cfg, source)
    return res[(res["entry_protocol"] == "unconditional") & (res["cost_level"] == "zero")]


class TestMartingaleWorld:
    def test_no_tradable_policy_has_edge(self, martingale_results):
        """5*SE family-wise bound: policy P&L distributions here are skewed
        mixtures (most exits on day 1), so small-n SEs understate tails --
        an n=24/4*SE version of this test false-alarmed on a seed fluke that
        an n=100 replication showed to be exactly zero (t=-0.02)."""
        for policy, sub in martingale_results.groupby("policy"):
            if policy == "oracle":
                continue
            pnls = sub["pnl"].to_numpy()
            se = pnls.std(ddof=1) / np.sqrt(len(pnls))
            assert abs(pnls.mean()) < 5 * max(se, 1e-12), policy

    def test_oracle_prints_money_from_noise(self, martingale_results):
        pnls = martingale_results[martingale_results["policy"] == "oracle"]["pnl"].to_numpy()
        se = pnls.std(ddof=1) / np.sqrt(len(pnls))
        assert pnls.mean() > 2 * se  # the quarantine exists for a reason


@pytest.fixture(scope="module")
def decay_results():
    cfg = load_config()
    cands = []
    for i in range(20):
        seed = 40_000 + 7 * i
        start = np.busday_offset(
            date(2026, 1, 5) + timedelta(weeks=i % 20), 0, roll="forward"
        ).item()
        half = 63
        a = simulate_gbm_paths(100.0, 0.05, 0.32, half / 252, half, 1, seed)[0]
        b = simulate_gbm_paths(float(a[-1]), 0.05, 0.10, half / 252, half, 1, seed + 1)[0]
        path = np.concatenate([a, b[1:]])
        quotes, spec = quotes_from_path(path, k=100.0, sigma_iv=0.20, r=R, start=start)
        cands.append(
            Candidate(
                quotes=quotes,
                spec=spec,
                pre_history=_pre_history(0.32, float(path[0]), start, seed + 2),
                scenario="edge_decay",
                seed=seed,
            )
        )
    res = run(cfg, _ListSource(cands))
    return res[res["entry_protocol"] == "unconditional"]


class TestDecayingEdgeWorld:
    """sigma_real: 0.32 for the first half, 0.10 after; marks stay at IV=0.20.
    Holding bleeds in the second half; a causal vol watcher should leave."""

    def test_causal_vol_watcher_beats_hold(self, decay_results):
        zero = decay_results[decay_results["cost_level"] == "zero"]
        m = zero.pivot_table(index="position_id", columns="policy", values="pnl")
        assert m["vol_regime_1.00"].mean() > m["hold_to_expiry"].mean()

    def test_positive_capture_where_detectable(self, decay_results):
        zero = decay_results[decay_results["cost_level"] == "zero"]
        cap = capture_fraction(zero, "vol_regime_1.00")
        assert cap > 0.1, cap  # the designed detector must collect real edge

    def test_vol_regime_exits_after_the_break_not_before(self, decay_results):
        zero = decay_results[
            (decay_results["cost_level"] == "zero")
            & (decay_results["policy"] == "vol_regime_1.00")
        ]
        # EWMA needs data to notice the break at life_frac 0.5; exits should
        # cluster after it but well before expiry
        med = zero["exit_frac"].median()
        assert 0.5 < med < 0.95, med

    def test_edge_survives_costs_directionally(self, decay_results):
        full = decay_results[decay_results["cost_level"] == "full_spread"]
        m = full.pivot_table(index="position_id", columns="policy", values="pnl")
        assert m["vol_regime_1.00"].mean() > m["hold_to_expiry"].mean()


class TestDeterminism:
    def test_same_seed_same_results(self):
        cfg = load_config()

        def one_run():
            source = SyntheticSource(
                scenarios=(Scenario("baseline", 0.06, 0.26, 0.20, n_days=42),),
                n_per_scenario=3,
                base_seed=9,
            )
            return run(cfg, source)

        pd.testing.assert_frame_equal(one_run(), one_run())


class TestDegenerateParameters:
    def test_fixed_time_one_equals_hold_exactly(self):
        cfg = load_config()
        source = SyntheticSource(
            scenarios=(Scenario("baseline", 0.06, 0.30, 0.20, n_days=42),),
            n_per_scenario=2,
            base_seed=17,
        )
        for cand in source.positions():
            rows = pd.DataFrame(
                run_candidate(cand, cfg, cfg.costs.levels[1], [HoldToExpiry(), FixedTime(1.0)])
            ).set_index("policy")
            assert rows.loc["fixed_time_1.00", "pnl"] == pytest.approx(
                rows.loc["hold_to_expiry", "pnl"], abs=1e-12
            )

    def test_puts_flow_end_to_end(self):
        cfg = load_config()
        source = SyntheticSource(
            scenarios=(
                Scenario("bear_put", mu=-0.15, sigma_real=0.30, sigma_iv=0.24,
                         moneyness=1.05, n_days=42, kind="put"),
            ),
            n_per_scenario=3,
            base_seed=23,
        )
        res = run(cfg, source)
        assert (res["kind"] == "put").all()
        assert np.isfinite(res["pnl"]).all()
        pivot = res[res["entry_protocol"] == "unconditional"].pivot_table(
            index=["position_id", "cost_level"], columns="policy", values="pnl"
        )
        assert (
            pivot["oracle"].to_numpy()[:, None]
            >= pivot.drop(columns="oracle").to_numpy() - 1e-9
        ).all()

    def test_two_day_position_survives_the_runner(self):
        cfg = load_config()
        path = np.array([100.0, 101.3])
        quotes, spec = quotes_from_path(path, k=100.0, sigma_iv=0.20, r=R)
        cand = Candidate(
            quotes=quotes, spec=spec,
            pre_history=_pre_history(0.2, 100.0, date(2026, 1, 5), 3),
            scenario="tiny", seed=3,
        )
        rows = pd.DataFrame(
            run_candidate(cand, cfg, cfg.costs.levels[0], [HoldToExpiry(), FixedTime(0.5)])
        )
        assert len(rows) == 3  # 2 policies + oracle
        assert (rows["exit_day"] == 1).all()  # only one executable day exists
