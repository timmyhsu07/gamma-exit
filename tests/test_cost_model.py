"""Transaction costs are first-class: the engine must charge every hedge trade
(entry, rebalances, final unwind) and P&L must degrade monotonically in cost."""

import numpy as np

from gamma_exit.pnl.engine import delta_hedge_synthetic, simulate_gbm_paths


def _run(cost):
    paths = simulate_gbm_paths(100.0, 0.05, 0.30, 1.0, 252, 1500, 7)
    return delta_hedge_synthetic(paths, 100.0, 1.0, 0.02, 0.20, cost_per_share=cost)


def test_zero_cost_charges_nothing():
    assert np.allclose(_run(0.0).trading_cost, 0.0)


def test_costs_are_positive_and_monotonic():
    free, half, full = _run(0.0), _run(0.005), _run(0.01)
    assert (half.trading_cost > 0).all()
    # doubling the per-share cost exactly doubles dollars paid (same trades)
    assert np.allclose(full.trading_cost, 2 * half.trading_cost)
    assert free.pnl.mean() > half.pnl.mean() > full.pnl.mean()


def test_pnl_gap_equals_compounded_costs_direction():
    """Costs reduce final P&L by at least the raw dollars paid (cash drag
    compounds at r, so the gap is >= the undiscounted cost)."""
    free, half = _run(0.0), _run(0.005)
    gap = free.pnl - half.pnl
    assert (gap >= half.trading_cost - 1e-9).all()
