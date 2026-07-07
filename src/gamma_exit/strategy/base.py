"""Exit-policy interface with causality enforced BY CONSTRUCTION (decision 4A).

A causal policy never touches a data frame. The backtest runner builds one
frozen `PositionState` per trading day containing only what a trader could
know at that day's close -- past-and-present marks, Greeks, own P&L, and the
CAUSAL vol forecast -- and the policy maps state -> HOLD | EXIT. Look-ahead
is not a bug a test catches; the future simply is not in scope.

Two consequences the tests pin down (tests/test_no_lookahead.py):
- building the state at day t from a truncated history yields an identical
  object, so no field can encode future information;
- `decide` must be a pure function of the state (deterministic, no memory
  between calls) -- policies with internal state would smuggle in path
  information the runner did not audit.

The oracle does NOT implement this interface. It lives quarantined in
strategy/oracle.py, consumes the full realized path, and is labeled
non-tradable everywhere it appears.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Decision(Enum):
    HOLD = "hold"
    EXIT = "exit"


@dataclass(frozen=True)
class PositionState:
    """Everything a trader knows at day t's close about one open position.

    All vols and times are trading-day basis (conventions.py). `forecast_vol`
    is the ONLY volatility a causal policy may consume; it is produced by
    vol/forecast.py from returns up to and including today. `float("nan")`
    means "no forecast available" -- policies must treat that as no-signal.
    """

    day_index: int  # 0 on entry day
    total_days: int  # trading sessions entry -> expiry (known at entry)
    date: datetime.date
    spot: float
    strike: float
    kind: Literal["call", "put"]
    expiry: datetime.date
    tte: float  # trading-years to expiry
    mark: float  # today's option mark (mid, or model if stale)
    stale: bool  # True when today's mark is model, not market
    tradable: bool  # True when a two-sided quote exists to exit into
    iv: float  # today's mark IV (ffilled through stale days)
    entry_iv: float
    entry_mark: float  # premium paid at entry (per share)
    delta: float
    gamma: float
    theta: float  # calendar theta, per trading-year
    vega: float
    cum_pnl: float  # X_t: realizable P&L so far, net of costs paid
    peak_pnl: float  # running max of cum_pnl up to today
    forecast_vol: float  # CAUSAL forecast at today's close (nan = no signal)
    r: float
    q: float

    @property
    def life_frac(self) -> float:
        """Fraction of the position's lifetime elapsed at today's close."""
        return self.day_index / self.total_days if self.total_days > 0 else 1.0


class ExitPolicy(ABC):
    """A causal exit rule: frozen state in, HOLD or EXIT out.

    The runner executes EXIT on the first *tradable* day >= the decision day
    (you cannot sell into a one-sided market). Decisions are taken at the
    close and executed at the same close -- the standard daily-backtest
    convention; its optimism is shared equally by every policy.
    """

    name: str = "abstract"

    @abstractmethod
    def decide(self, state: PositionState) -> Decision: ...
