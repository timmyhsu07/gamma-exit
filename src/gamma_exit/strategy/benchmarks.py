"""Benchmark exit rules: the floor every causal policy must beat.

Both are causal (trivially -- they use no market data at all), so they share
the PositionState interface.
"""

from __future__ import annotations

from gamma_exit.strategy.base import Decision, ExitPolicy, PositionState


class HoldToExpiry(ExitPolicy):
    """The paper's baseline: never exit; take settlement."""

    name = "hold_to_expiry"

    def decide(self, state: PositionState) -> Decision:
        return Decision.HOLD


class FixedTime(ExitPolicy):
    """Exit once a fixed fraction of the position's lifetime has elapsed.

    The paper's 'Fixed-Time' benchmark uses the mean of the ORACLE's stop
    times (~0.79 of lifetime in its Table 3) -- information a trader would
    not have, but a defensible static rule to benchmark against. The
    fraction is a constructor parameter so it can also be fit on a training
    split and apply it out-of-sample (walk-forward honest version).
    """

    def __init__(self, fraction: float = 0.79) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self.name = f"fixed_time_{fraction:.2f}"

    def decide(self, state: PositionState) -> Decision:
        return Decision.EXIT if state.life_frac >= self.fraction else Decision.HOLD
