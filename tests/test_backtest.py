"""Milestone 5 gate: exit accounting is exact, the oracle is a true ceiling,
the runner produces a coherent results panel, and metrics behave.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from _synthetic_chain import synthetic_quotes
from gamma_exit.analytics.metrics import capture_fraction, summarize
from gamma_exit.analytics.regimes import summarize_by_regime, tag_regimes
from gamma_exit.backtest.runner import run
from gamma_exit.backtest.synthetic import Scenario, SyntheticSource
from gamma_exit.config import load_config
from gamma_exit.pnl.replay import exit_values, replay_position

R = 0.02


class TestExitValueAccounting:
    """exit_values() must equal an EXPLICIT replay that exits on that day --
    the strongest possible check on the cost adjustments."""

    @pytest.mark.parametrize("frac,share_cost", [(0.0, 0.0), (0.5, 0.005), (1.0, 0.01)])
    def test_matches_explicit_early_exit_replay(self, frac, share_cost):
        quotes, spec, _ = synthetic_quotes(n_days=80, spread=0.10, seed=33)
        hold = replay_position(
            quotes, spec, R, share_cost_per_share=share_cost, option_spread_frac=frac
        )
        values, executable = exit_values(hold)
        for e in (1, 20, 55, 79):
            assert executable[e]
            early = replay_position(
                quotes,
                replace(spec, exit_date=quotes["date"].iloc[e]),
                R,
                share_cost_per_share=share_cost,
                option_spread_frac=frac,
            )
            want = early.pnl if e < 80 else hold.pnl
            assert values[e] == pytest.approx(want, abs=1e-9), e

    def test_day_zero_never_executable(self):
        quotes, spec, _ = synthetic_quotes(n_days=30, seed=3)
        _, executable = exit_values(replay_position(quotes, spec, R))
        assert not executable[0]
        assert executable[-1]  # settlement always available

    def test_stale_days_not_executable(self):
        quotes, spec, _ = synthetic_quotes(n_days=30, seed=3)
        quotes.loc[10, ["bid", "ask", "mid"]] = np.nan
        values, executable = exit_values(replay_position(quotes, spec, R))
        assert not executable[10]
        assert np.isnan(values[10])


@pytest.fixture(scope="module")
def results():
    cfg = load_config()
    source = SyntheticSource(
        scenarios=(
            Scenario("baseline", mu=0.06, sigma_real=0.26, sigma_iv=0.20, n_days=42),
            Scenario("bear", mu=-0.20, sigma_real=0.30, sigma_iv=0.26, n_days=42),
        ),
        n_per_scenario=6,
        base_seed=11,
    )
    return run(cfg, source)


class TestRunnerPanel:

    def test_panel_is_complete(self, results):
        cfg = load_config()
        uncond = results[results["entry_protocol"] == "unconditional"]
        n_pos = uncond["position_id"].nunique()
        n_policies = uncond["policy"].nunique()  # 5 causal/benchmark + oracle
        n_costs = uncond["cost_level"].nunique()
        assert n_pos == 12
        assert n_policies == len(cfg.policies)  # oracle included in count
        assert n_costs == 3
        assert len(uncond) == n_pos * n_policies * n_costs

    def test_oracle_dominates_and_costs_hurt(self, results):
        uncond = results[results["entry_protocol"] == "unconditional"]
        # oracle >= every policy, position by position, at each cost level
        pivot = uncond.pivot_table(
            index=["position_id", "cost_level"], columns="policy", values="pnl"
        )
        others = pivot.drop(columns="oracle")
        assert (pivot["oracle"].to_numpy()[:, None] >= others.to_numpy() - 1e-9).all()
        # and zero-cost mean beats full-spread mean for every policy
        by_cost = uncond.pivot_table(index="policy", columns="cost_level", values="pnl")
        assert (by_cost["zero"] >= by_cost["full_spread"] - 1e-12).all()

    def test_hold_to_expiry_exits_at_one(self, results):
        hold = results[results["policy"] == "hold_to_expiry"]
        assert (hold["exit_frac"] == 1.0).all()

    def test_forecast_gate_is_a_subset(self, results):
        uncond = set(results[results["entry_protocol"] == "unconditional"]["position_id"])
        gated = set(results[results["entry_protocol"] == "forecast_rv_gt_iv"]["position_id"])
        assert gated <= uncond

    def test_metrics_summary_shape_and_capture(self, results):
        uncond = results[results["entry_protocol"] == "unconditional"]
        table = summarize(uncond)
        assert {"mean_pnl", "win_rate", "capture"} <= set(table.columns)
        # capture is 1.0 for the oracle itself, by definition
        for cl in ("zero", "half_spread", "full_spread"):
            assert capture_fraction(
                uncond[uncond["cost_level"] == cl], "oracle"
            ) == pytest.approx(1.0)

    def test_regime_tagging(self, results):
        tagged = tag_regimes(results)
        assert set(tagged["vol_regime"]) <= {"rv_above_iv", "rv_below_iv"}
        assert set(tagged["drift_regime"]) <= {"bull", "bear", "flat"}
        # bear scenario windows must be dominated by bear/flat tags
        bear = tagged[tagged["scenario"] == "bear"]
        assert (bear["drift_regime"] != "bull").mean() > 0.6
        table = summarize_by_regime(tagged, "vol_regime")
        assert "oracle" in table.columns

    def test_results_artifact_written(self, tmp_path):
        cfg = load_config()
        source = SyntheticSource(
            scenarios=(Scenario("baseline", 0.06, 0.26, 0.20, n_days=21),),
            n_per_scenario=2,
            base_seed=5,
        )
        run(cfg, source, out_dir=tmp_path)
        parquets = list(tmp_path.glob("results_*.parquet"))
        metas = list(tmp_path.glob("results_*.meta.json"))
        assert len(parquets) == 1 and len(metas) == 1
        meta = pd.read_json(metas[0], typ="series")
        assert "git_commit" in meta and "config" in meta


class TestCaptureEdgeCases:
    def _frame(self, hold, oracle, policy):
        rows = []
        for i, (h, o, p) in enumerate(zip(hold, oracle, policy)):
            for name, pnl in (("hold_to_expiry", h), ("oracle", o), ("mypol", p)):
                rows.append(
                    {
                        "position_id": f"p{i}",
                        "entry_date": f"2026-01-{(i % 9) + 1:02d}",
                        "policy": name,
                        "pnl": pnl,
                        "cost_level": "zero",
                        "exit_frac": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_perfect_capture_is_one(self):
        df = self._frame(hold=[0, 0], oracle=[2, 4], policy=[2, 4])
        assert capture_fraction(df, "mypol") == pytest.approx(1.0)

    def test_no_oracle_edge_is_nan(self):
        df = self._frame(hold=[3, 3], oracle=[3, 3], policy=[1, 1])
        assert np.isnan(capture_fraction(df, "mypol"))

    def test_half_capture(self):
        df = self._frame(hold=[0, 0], oracle=[4, 4], policy=[2, 2])
        assert capture_fraction(df, "mypol") == pytest.approx(0.5)
