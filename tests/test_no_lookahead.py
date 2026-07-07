"""HARD RULE: causal policies can never see the future.

Causality is enforced by construction -- policies receive a frozen
PositionState built by the runner, never a data frame -- so these tests pin
the construction:

1. the state at day t built from a TRUNCATED history equals the state built
   from the full history (no field can encode rows > t);
2. the causal forecast series is prefix-invariant;
3. policy decisions are pure functions of the state;
4. the oracle is quarantined: not an ExitPolicy, not registrable, and its
   input type (a full exit-value array) is never handed to policies.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from _synthetic_chain import synthetic_quotes
from gamma_exit.backtest.runner import build_states, causal_forecast_series
from gamma_exit.backtest.synthetic import PRE_HISTORY_DAYS, SyntheticSource
from gamma_exit.pnl.replay import replay_position
from gamma_exit.strategy import ExitPolicy, make_policies
from gamma_exit.strategy.oracle import oracle_exit

R, Q = 0.02, 0.0
N_DAYS = 60


def _position():
    quotes, spec, _ = synthetic_quotes(n_days=N_DAYS, sigma_real=0.28, seed=77)
    pre_dates = pd.bdate_range(end=spec.entry_date, periods=PRE_HISTORY_DAYS + 1).date[:-1]
    rng = np.random.default_rng(5)
    pre = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.28 / np.sqrt(252), PRE_HISTORY_DAYS))),
        index=pre_dates,
    )
    return quotes, spec, pre


def _states(quotes, spec, pre):
    res = replay_position(quotes, spec, R, q=Q)
    spots = pd.Series(res.daily["spot"].to_numpy(), index=res.daily["date"].to_numpy())
    forecast = causal_forecast_series(pre, spots, decay=0.94)
    return build_states(res, R, Q, forecast)


class TestStateTruncationInvariance:
    def test_state_at_t_identical_under_truncation(self):
        quotes, spec, pre = _position()
        full_states = _states(quotes, spec, pre)
        for t in (1, 10, 37):
            # rebuild the world as it existed at day t: only quotes <= t,
            # with the position force-exited at that day's close
            trunc_quotes = quotes.iloc[: t + 1]
            trunc_spec = replace(spec, exit_date=quotes["date"].iloc[t])
            trunc_states = _states(trunc_quotes, trunc_spec, pre)
            got, want = trunc_states[t], full_states[t]
            for field in (
                "date spot strike kind expiry tte mark iv entry_iv entry_mark "
                "delta gamma theta vega cum_pnl peak_pnl forecast_vol stale"
            ).split():
                g, w = getattr(got, field), getattr(want, field)
                if isinstance(g, float) and np.isnan(g):
                    assert np.isnan(w), field
                else:
                    assert g == pytest.approx(w, abs=1e-12), field

    def test_forecast_series_is_prefix_invariant(self):
        quotes, spec, pre = _position()
        spots = pd.Series(quotes["spot"].to_numpy(), index=quotes["date"].to_numpy())
        full = causal_forecast_series(pre, spots, decay=0.94)
        for t in (5, 30):
            trunc = causal_forecast_series(pre, spots.iloc[: t + 1], decay=0.94)
            assert float(trunc.iloc[t]) == pytest.approx(float(full.iloc[t]), rel=1e-12)


class TestPolicyPurity:
    def test_decisions_deterministic_and_stateless(self):
        quotes, spec, pre = _position()
        states = _states(quotes, spec, pre)
        for pol in make_policies(
            ["hold_to_expiry", "fixed_time", "theta_gamma_threshold",
             "trailing_stop", "vol_regime"]
        ):
            first = [pol.decide(s) for s in states]
            second = [pol.decide(s) for s in reversed(states)][::-1]
            assert first == second, pol.name  # no internal state, no memory


class TestOracleQuarantine:
    def test_oracle_is_not_an_exit_policy(self):
        import inspect

        from gamma_exit.strategy import oracle as oracle_module

        assert not any(
            isinstance(obj, type) and issubclass(obj, ExitPolicy)
            for obj in vars(oracle_module).values()
            if inspect.isclass(obj)
        ), "oracle module must not define an ExitPolicy"

    def test_oracle_not_constructible_from_registry(self):
        assert make_policies(["oracle"]) == []  # skipped, never instantiated
        with pytest.raises(ValueError, match="unknown policy"):
            make_policies(["oracle_exit"])

    def test_oracle_labeled_non_tradable(self):
        from gamma_exit.strategy import oracle as oracle_module

        assert "NON-TRADABLE" in (oracle_module.__doc__ or "")

    def test_oracle_dominates_every_policy_pathwise(self):
        """The quarantined ceiling must actually BE a ceiling."""
        from gamma_exit.config import load_config
        from gamma_exit.backtest.runner import run_candidate
        from gamma_exit.strategy import make_policies as mk

        cfg = load_config()
        source = SyntheticSource(n_per_scenario=2, base_seed=1)
        pols = mk(cfg.policies)
        for cand in source.positions()[:6]:
            for cost in cfg.costs.levels:
                rows = pd.DataFrame(run_candidate(cand, cfg, cost, pols))
                oracle_pnl = rows.loc[rows["policy"] == "oracle", "pnl"].iloc[0]
                others = rows.loc[rows["policy"] != "oracle", "pnl"]
                assert (oracle_pnl >= others - 1e-9).all()


class TestOracleInputShape:
    def test_requires_executable_day(self):
        with pytest.raises(ValueError, match="no executable"):
            oracle_exit(np.zeros(4), np.array([True, False, False, False]))

    def test_never_exits_at_entry(self):
        values = np.array([99.0, 1.0, 2.0, 3.0])
        executable = np.array([True, True, True, True])
        day, pnl = oracle_exit(values, executable)
        assert day == 3 and pnl == 3.0
