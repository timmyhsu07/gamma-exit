"""Exit policies (Milestone 4).

- base.py        frozen PositionState + ExitPolicy ABC (causality by
                 construction: the state object contains no future data)
- benchmarks.py  HoldToExpiry, FixedTime
- causal.py      ThetaGammaThreshold, TrailingStop, VolRegime
- oracle.py      QUARANTINED ex-post argmax -- non-causal, non-tradable
                 ceiling; deliberately NOT an ExitPolicy and NOT in the
                 registry below.

`make_policies` maps config policy names to instances with their default
parameters; parameterized variants use explicit specs (e.g.
"fixed_time:0.5"). "oracle" in a config's policy list is handled by the
runner directly -- it is not constructible here, on purpose.
"""

from __future__ import annotations

from gamma_exit.strategy.base import Decision, ExitPolicy, PositionState
from gamma_exit.strategy.benchmarks import FixedTime, HoldToExpiry
from gamma_exit.strategy.causal import ThetaGammaThreshold, TrailingStop, VolRegime

__all__ = [
    "Decision",
    "ExitPolicy",
    "PositionState",
    "HoldToExpiry",
    "FixedTime",
    "ThetaGammaThreshold",
    "TrailingStop",
    "VolRegime",
    "make_policies",
]

_FACTORIES = {
    "hold_to_expiry": lambda arg: HoldToExpiry(),
    "fixed_time": lambda arg: FixedTime(float(arg)) if arg else FixedTime(),
    "theta_gamma_threshold": lambda arg: (
        ThetaGammaThreshold(float(arg)) if arg else ThetaGammaThreshold()
    ),
    "trailing_stop": lambda arg: TrailingStop(float(arg)) if arg else TrailingStop(),
    "vol_regime": lambda arg: VolRegime(float(arg)) if arg else VolRegime(),
}


def make_policies(names: list[str]) -> list[ExitPolicy]:
    """Instantiate causal policies from config specs ("name" or "name:param").

    "oracle" is skipped here: the runner computes it through the quarantined
    path so it can never be mistaken for a tradable rule.
    """
    policies = []
    for spec in names:
        if spec == "oracle":
            continue
        base, _, arg = spec.partition(":")
        if base not in _FACTORIES:
            raise ValueError(f"unknown policy {spec!r}; known: {sorted(_FACTORIES)}")
        policies.append(_FACTORIES[base](arg))
    return policies
