"""Milestone 3 gate: the real-data replay pipeline must reproduce the
Milestone 1 synthetic engine EXACTLY when fed a synthetic chain in canonical
quote format (loader shape -> per-day IV solve -> deltas -> shared core).
I treat this as the gate for trusting anything replay-mode ever outputs.

If these pass, replay-mode results inherit the M1 validation: the only new
trust assumptions on real data are the quotes themselves.
"""

from dataclasses import replace
from datetime import date, datetime, timezone

import numpy as np
import pytest

from _synthetic_chain import synthetic_quotes
from gamma_exit.data.cache import WriteOnceCache
from gamma_exit.data.schema import OptionRecord, options_to_frame
from gamma_exit.pnl.engine import delta_hedge_synthetic
from gamma_exit.pnl.replay import PositionSpec, load_option_history, replay_position

R = 0.02
N_DAYS = 126
T_YEARS = N_DAYS / 252


class TestEquivalenceGate:
    """Replay(synthetic chain) == synthetic engine, pathwise, to solver tol."""

    def test_pnl_path_matches_synthetic_engine(self):
        quotes, spec, paths = synthetic_quotes(n_days=N_DAYS, seed=314)
        rep = replay_position(quotes, spec, R)
        syn = delta_hedge_synthetic(paths, spec.strike, T_YEARS, R, 0.20)
        assert rep.pnl == pytest.approx(float(syn.pnl[0]), abs=1e-5)
        np.testing.assert_allclose(
            rep.daily["cum_net"].to_numpy(), syn.pnl_path[0], atol=1e-5
        )

    def test_equivalence_holds_with_dividend_yield(self):
        quotes, spec, paths = synthetic_quotes(n_days=N_DAYS, q=0.012, seed=99)
        rep = replay_position(quotes, spec, R, q=0.012)
        syn = delta_hedge_synthetic(paths, spec.strike, T_YEARS, R, 0.20, q=0.012)
        assert rep.pnl == pytest.approx(float(syn.pnl[0]), abs=1e-5)

    def test_equivalence_for_puts(self):
        quotes, spec, paths = synthetic_quotes(n_days=N_DAYS, kind="put", seed=7)
        rep = replay_position(quotes, spec, R)
        syn = delta_hedge_synthetic(paths, spec.strike, T_YEARS, R, 0.20, kind="put")
        assert rep.pnl == pytest.approx(float(syn.pnl[0]), abs=1e-5)

    def test_share_costs_flow_through_identically(self):
        quotes, spec, paths = synthetic_quotes(n_days=N_DAYS, seed=314)
        rep = replay_position(quotes, spec, R, share_cost_per_share=0.005)
        syn = delta_hedge_synthetic(
            paths, spec.strike, T_YEARS, R, 0.20, cost_per_share=0.005
        )
        assert rep.pnl == pytest.approx(float(syn.pnl[0]), abs=1e-5)
        assert rep.trading_cost == pytest.approx(float(syn.trading_cost[0]), abs=1e-6)


class TestAttribution:
    def test_components_sum_to_net_exactly_and_cum_matches(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, seed=42)
        rep = replay_position(
            quotes, spec, R, share_cost_per_share=0.005, option_spread_frac=0.5
        )
        d = rep.daily
        recon = d["gamma_pnl"] + d["theta_pnl"] + d["vega_pnl"] + d["carry"] - d["cost"] + d["residual"]
        np.testing.assert_allclose(recon, d["net"], atol=1e-12)
        assert d["net"].sum() == pytest.approx(rep.pnl, abs=1e-9)
        assert d["cum_net"].iloc[-1] == pytest.approx(rep.pnl, abs=1e-12)

    def test_residual_is_small_on_smooth_marks(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, seed=42)
        d = replay_position(quotes, spec, R).daily
        # residual = higher-order terms only; tiny next to the gamma/theta flows
        assert d["residual"].abs().sum() < 0.10 * (
            d["gamma_pnl"].abs().sum() + d["theta_pnl"].abs().sum()
        )

    def test_gamma_positive_theta_negative_for_long_option(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, seed=42)
        d = replay_position(quotes, spec, R).daily
        assert (d["gamma_pnl"].iloc[1:] >= 0).all()
        assert (d["theta_pnl"].iloc[1:] < 0).all()

    def test_vega_pnl_captures_a_marking_vol_jump(self):
        iv = np.full(N_DAYS + 1, 0.20)
        iv[60:] = 0.26
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, sigma_iv=iv, seed=11)
        d = replay_position(quotes, spec, R).daily
        jump = d.iloc[60]
        assert jump["vega_pnl"] > 0
        # vega term explains most of the jump-day mark change beyond gamma/theta
        assert abs(jump["residual"]) < 0.2 * jump["vega_pnl"]
        # constant after the jump (up to IV-solver tolerance noise)
        assert d["vega_pnl"].iloc[61:].abs().max() < 1e-8


class TestOptionSpreadCosts:
    def test_exit_before_expiry_pays_both_sides_exactly(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, spread=0.10, seed=21)
        early = replace(spec, exit_date=quotes["date"].iloc[80])
        free = replay_position(quotes, early, R)
        half = replay_position(quotes, early, R, option_spread_frac=0.5)
        full = replay_position(quotes, early, R, option_spread_frac=1.0)
        hs = 0.05  # half-spread of the synthetic chain
        assert free.pnl - half.pnl == pytest.approx(0.5 * 2 * hs, abs=1e-12)
        assert free.pnl - full.pnl == pytest.approx(1.0 * 2 * hs, abs=1e-12)
        assert full.trading_cost == pytest.approx(2 * hs, abs=1e-12)

    def test_expiry_settlement_pays_entry_side_only(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, spread=0.10, seed=21)
        free = replay_position(quotes, spec, R)
        full = replay_position(quotes, spec, R, option_spread_frac=1.0)
        assert free.pnl - full.pnl == pytest.approx(0.05, abs=1e-12)


class TestQuoteHygiene:
    def test_stale_days_marked_to_model_and_flagged(self):
        quotes, spec, paths = synthetic_quotes(n_days=N_DAYS, seed=314)
        quotes.loc[30:34, ["bid", "ask", "mid"]] = np.nan
        rep = replay_position(quotes, spec, R)
        assert rep.summary["stale_days"] == 5
        assert rep.daily["stale"].iloc[30:35].all()
        assert rep.daily["vega_pnl"].iloc[30:35].abs().max() == 0.0  # IV ffill
        # constant marking vol => model mark == the missing mid, so the
        # equivalence with the synthetic engine survives the gap
        syn = delta_hedge_synthetic(paths, spec.strike, T_YEARS, R, 0.20)
        assert rep.pnl == pytest.approx(float(syn.pnl[0]), abs=1e-5)

    def test_mid_or_drop_at_entry_and_exit(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, seed=1)
        bad_entry = quotes.copy()
        bad_entry.loc[0, ["bid", "ask", "mid"]] = np.nan
        with pytest.raises(ValueError, match="mid-or-drop"):
            replay_position(bad_entry, spec, R)

        early = replace(spec, exit_date=quotes["date"].iloc[50])
        bad_exit = quotes.copy()
        bad_exit.loc[50, ["bid", "ask", "mid"]] = np.nan
        with pytest.raises(ValueError, match="mid-or-drop"):
            replay_position(bad_exit, early, R)

    def test_missing_entry_or_exit_row_fails_loudly(self):
        quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, seed=1)
        with pytest.raises(ValueError, match="entry date"):
            replay_position(quotes.iloc[3:], spec, R)
        gap = quotes[quotes["date"] != spec.expiry]
        with pytest.raises(ValueError, match="exit date"):
            replay_position(gap, spec, R)

    def test_exit_after_expiry_rejected(self):
        quotes, spec, _ = synthetic_quotes(n_days=20, seed=1)
        with pytest.raises(ValueError, match="after expiry"):
            replay_position(quotes, replace(spec, exit_date=date(2027, 1, 1)), R)


class TestLoader:
    ASOF_1 = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)  # Wed 16:00 ET
    ASOF_1B = datetime(2026, 7, 1, 21, 0, tzinfo=timezone.utc)  # later same day
    ASOF_2 = datetime(2026, 7, 2, 20, 0, tzinfo=timezone.utc)  # Thu
    ASOF_SAT = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)  # Saturday pull

    def _rec(self, asof, strike=100.0, kind="call", bid=1.0, ask=1.2, spot=101.0):
        return OptionRecord(
            asof=asof, provider="test", underlying="SYN", expiry=date(2026, 8, 21),
            strike=strike, kind=kind, bid=bid, ask=ask, volume=100,
            open_interest=500, underlying_price=spot,
        )

    def test_dedupes_filters_and_drops_weekend_pulls(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        cache.write(
            options_to_frame(
                [
                    self._rec(self.ASOF_1, bid=0.9, ask=1.1),
                    self._rec(self.ASOF_1, strike=110.0),  # other strike: filtered
                    self._rec(self.ASOF_1, kind="put"),  # other kind: filtered
                ]
            ),
            "chains", "SYN/pull1",
        )
        cache.write(  # later pull the same day must win the dedupe
            options_to_frame([self._rec(self.ASOF_1B, bid=1.0, ask=1.2, spot=102.0)]),
            "chains", "SYN/pull1b",
        )
        cache.write(options_to_frame([self._rec(self.ASOF_2, spot=103.0)]), "chains", "SYN/p2")
        cache.write(options_to_frame([self._rec(self.ASOF_SAT)]), "chains", "SYN/sat")

        spec = PositionSpec(
            underlying="SYN", expiry=date(2026, 8, 21), strike=100.0, kind="call",
            entry_date=date(2026, 7, 1),
        )
        q = load_option_history(cache, spec)
        assert q["date"].tolist() == [date(2026, 7, 1), date(2026, 7, 2)]
        assert q["spot"].tolist() == [102.0, 103.0]  # 21:00 pull won day one
        assert q["mid"].iloc[0] == pytest.approx(1.1)

    def test_unknown_contract_fails_loudly(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        cache.write(options_to_frame([self._rec(self.ASOF_1)]), "chains", "SYN/p1")
        spec = PositionSpec(
            underlying="SYN", expiry=date(2026, 8, 21), strike=999.0, kind="call",
            entry_date=date(2026, 7, 1),
        )
        with pytest.raises(ValueError, match="no cached quotes"):
            load_option_history(cache, spec)
