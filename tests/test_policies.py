"""Unit tests for exit policies: each rule fires exactly when its docstring
says it does, on hand-built states."""

import math
from datetime import date

import pytest

from gamma_exit.strategy import (
    Decision,
    FixedTime,
    HoldToExpiry,
    PositionState,
    ThetaGammaThreshold,
    TrailingStop,
    VolRegime,
    make_policies,
)


def make_state(**overrides) -> PositionState:
    base = dict(
        day_index=10,
        total_days=60,
        date=date(2026, 3, 2),
        spot=100.0,
        strike=100.0,
        kind="call",
        expiry=date(2026, 5, 29),
        tte=50 / 252,
        mark=4.0,
        stale=False,
        tradable=True,
        iv=0.20,
        entry_iv=0.20,
        entry_mark=5.0,
        delta=0.5,
        gamma=0.04,
        theta=-12.0,
        vega=18.0,
        cum_pnl=1.0,
        peak_pnl=1.5,
        forecast_vol=0.25,
        r=0.02,
        q=0.0,
    )
    return PositionState(**{**base, **overrides})


class TestBenchmarks:
    def test_hold_to_expiry_never_exits(self):
        pol = HoldToExpiry()
        for t in range(0, 61, 10):
            assert pol.decide(make_state(day_index=t)) is Decision.HOLD

    def test_fixed_time_exits_at_fraction(self):
        pol = FixedTime(0.5)
        assert pol.decide(make_state(day_index=29)) is Decision.HOLD
        assert pol.decide(make_state(day_index=30)) is Decision.EXIT
        assert pol.decide(make_state(day_index=59)) is Decision.EXIT

    def test_fixed_time_validates_fraction(self):
        with pytest.raises(ValueError):
            FixedTime(0.0)
        with pytest.raises(ValueError):
            FixedTime(1.5)


class TestThetaGammaThreshold:
    def test_threshold_boundary(self):
        pol = ThetaGammaThreshold(1.0)
        # income = 0.5 * 0.04 * 100^2 * 0.25^2 = 12.5 > rent |theta| = 12 -> HOLD
        assert pol.decide(make_state()) is Decision.HOLD
        # forecast collapses -> income 0.5*0.04*1e4*0.15^2 = 4.5 < 12 -> EXIT
        assert pol.decide(make_state(forecast_vol=0.15)) is Decision.EXIT
        # a higher bar exits even at the original forecast: 12.5/12 < 1.5
        assert ThetaGammaThreshold(1.5).decide(make_state()) is Decision.EXIT

    def test_no_forecast_means_hold(self):
        pol = ThetaGammaThreshold(1.0)
        assert pol.decide(make_state(forecast_vol=math.nan)) is Decision.HOLD

    def test_zero_theta_means_hold(self):
        pol = ThetaGammaThreshold(1.0)
        assert pol.decide(make_state(theta=0.0, forecast_vol=0.01)) is Decision.HOLD


class TestTrailingStop:
    def test_exits_on_drawdown_from_peak(self):
        pol = TrailingStop(drawdown_frac=0.5)  # budget = 0.5 * entry_mark 5 = 2.5
        assert pol.decide(make_state(peak_pnl=3.0, cum_pnl=1.0)) is Decision.HOLD
        assert pol.decide(make_state(peak_pnl=3.5, cum_pnl=1.0)) is Decision.EXIT

    def test_pure_loss_also_triggers(self):
        pol = TrailingStop(drawdown_frac=0.5)
        assert pol.decide(make_state(peak_pnl=0.0, cum_pnl=-2.6)) is Decision.EXIT


class TestVolRegime:
    def test_exits_when_forecast_below_entry_iv(self):
        pol = VolRegime(1.0)
        assert pol.decide(make_state(forecast_vol=0.21)) is Decision.HOLD
        assert pol.decide(make_state(forecast_vol=0.19)) is Decision.EXIT

    def test_no_forecast_means_hold(self):
        pol = VolRegime(1.0)
        assert pol.decide(make_state(forecast_vol=math.nan)) is Decision.HOLD


class TestRegistry:
    def test_config_names_construct_defaults(self):
        pols = make_policies(
            ["hold_to_expiry", "fixed_time", "oracle", "theta_gamma_threshold",
             "trailing_stop", "vol_regime"]
        )
        names = [p.name for p in pols]
        assert "oracle" not in names and len(pols) == 5

    def test_parameterized_specs(self):
        (pol,) = make_policies(["fixed_time:0.5"])
        assert pol.name == "fixed_time_0.50"

    def test_unknown_name_fails(self):
        with pytest.raises(ValueError, match="unknown policy"):
            make_policies(["martingale"])
