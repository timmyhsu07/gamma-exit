"""Causal exit policies -- the research contribution lives here.

Each rule consumes ONLY the frozen PositionState (past P&L, today's Greeks,
and the causal forecast vol). The question the backtest answers: what fraction of
the oracle's (non-tradable) edge over hold-to-expiry do these capture, net
of costs?

A missing forecast (nan) is a no-signal day: vol-based rules HOLD rather
than act on data they do not have.
"""

from __future__ import annotations

import math

from gamma_exit.strategy.base import Decision, ExitPolicy, PositionState


class ThetaGammaThreshold(ExitPolicy):
    """Exit when expected gamma income no longer pays the theta rent.

    Expected daily P&L of the hedged position under the CAUSAL forecast:

        E[dPnL] ~ 1/2 * Gamma * S^2 * sigma_forecast^2 * dt + Theta * dt

    (Theta is negative for a long option.) Exit when the expected-income to
    rent ratio drops below `threshold`:

        (1/2 * Gamma * S^2 * sigma_f^2) / |Theta| < threshold

    threshold = 1.0 is the break-even point of the gamma-scalping/theta
    trade-off itself; > 1 demands a margin of edge to keep paying the rent.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.threshold = threshold
        self.name = f"theta_gamma_{threshold:.2f}"

    def decide(self, state: PositionState) -> Decision:
        if math.isnan(state.forecast_vol):
            return Decision.HOLD
        rent = abs(state.theta)
        if rent < 1e-12:  # no decay to pay (deep ITM/OTM long-dated): hold
            return Decision.HOLD
        income = 0.5 * state.gamma * state.spot**2 * state.forecast_vol**2
        return Decision.EXIT if income / rent < self.threshold else Decision.HOLD


class TrailingStop(ExitPolicy):
    """Exit after giving back `drawdown_frac` of entry premium from the peak.

    Pure past-P&L rule (no vol input): protects harvested gamma gains from
    the accelerating late-life theta bleed. Drawdown is measured in dollars
    as a fraction of the entry mark so the rule scales across underlyings.
    """

    def __init__(self, drawdown_frac: float = 0.5, entry_mark_floor: float = 1e-6) -> None:
        if drawdown_frac <= 0:
            raise ValueError(f"drawdown_frac must be positive, got {drawdown_frac}")
        self.drawdown_frac = drawdown_frac
        self.entry_mark_floor = entry_mark_floor
        self.name = f"trailing_stop_{drawdown_frac:.2f}"

    def decide(self, state: PositionState) -> Decision:
        budget = self.drawdown_frac * max(state.entry_mark, self.entry_mark_floor)
        return Decision.EXIT if state.peak_pnl - state.cum_pnl >= budget else Decision.HOLD


class VolRegime(ExitPolicy):
    """Exit when the causal forecast says the vol edge is gone.

    A delta-hedged long option is long realized-vs-implied variance; the
    position's premise dies when forecast vol falls to `ratio` times the
    ENTRY implied vol (the vol that was paid for).
    """

    def __init__(self, ratio: float = 1.0) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        self.ratio = ratio
        self.name = f"vol_regime_{ratio:.2f}"

    def decide(self, state: PositionState) -> Decision:
        if math.isnan(state.forecast_vol):
            return Decision.HOLD
        return (
            Decision.EXIT
            if state.forecast_vol < self.ratio * state.entry_iv
            else Decision.HOLD
        )
