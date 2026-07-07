"""=======================  NON-CAUSAL / NON-TRADABLE  =======================

THE ORACLE. This module sees the FUTURE. Nothing in here is a strategy.

The oracle exit is the ex-post argmax of realizable position value over the
whole holding window -- the paper's "Optimal Stopping". It requires knowing
the entire realized path at decision time, which no trader has. It exists
for exactly one purpose: an UPPER BOUND on what any exit rule could have
earned, so causal policies can be scored as a fraction of it:

    capture = (policy - hold_to_expiry) / (oracle - hold_to_expiry)

Quarantine rules (enforced by tests/test_no_lookahead.py):
- `oracle_exit` does NOT implement the ExitPolicy interface and cannot be
  registered as a policy; it consumes a full exit-value array, a type of
  input the runner never hands to causal policies.
- Results derived from it are labeled "oracle" and reported as a ceiling,
  never in the same breath as tradable performance.
==========================================================================="""

from __future__ import annotations

import numpy as np


def oracle_exit(exit_values: np.ndarray, executable: np.ndarray) -> tuple[int, float]:
    """Ex-post best exit: argmax of realizable value over executable days.

    exit_values[j]: realizable P&L exiting at day j (cost-adjusted);
    executable[j]: whether an exit could actually trade at day j (two-sided
    quote, or settlement on the final day). Day 0 is never executable (the
    position was just entered at that close).

    Returns (day_index, pnl). By construction this dominates every policy
    that exits on an executable day -- including hold-to-expiry -- at the
    same cost level; tests assert that dominance.
    """
    if exit_values.shape != executable.shape:
        raise ValueError("exit_values and executable must share one shape")
    if not executable[1:].any():
        raise ValueError("no executable exit day in the window")
    candidates = np.where(executable)[0]
    candidates = candidates[candidates > 0]
    best = candidates[np.argmax(exit_values[candidates])]
    return int(best), float(exit_values[best])
