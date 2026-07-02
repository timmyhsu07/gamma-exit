"""Synthetic delta-hedged P&L engine and the gamma-vega identity integral.

The trade being simulated
-------------------------
At t=0: buy one European option priced at Black-Scholes with sigma_iv, short
delta_0 shares, park the net proceeds in a money-market account accruing r.
At each rehedge date: mark the option at BS(sigma_iv), reset the share hedge
to the current BS delta, and let cash absorb the trades. The portfolio is
self-financing and starts at exactly zero value.

The identity being tested (Milestone 1 gate)
--------------------------------------------
With the true path following dS = mu S dt + sigma_real S dW and the hedge run
at sigma_iv, continuous-time self-financing accounting gives, PATHWISE:

    X_T = integral_0^T e^{r (T-u)} * 1/2 * Gamma_iv(u, S_u) * S_u^2
              * (sigma_real^2 - sigma_iv^2) du

Two things matter and are asserted in tests/test_pnl_identity.py:
1. Discrete-hedge P&L converges to this integral as rehedge frequency rises.
2. The drift mu does NOT appear. The paper's extra -(mu - r) S^2 Gamma term
   is an artifact of its non-self-financing accounting; we test empirically
   that no drift bias exists (see PROJECT_BRIEF.md section 1.3).

Transaction costs enter as a proportional cost per share traded
(cost_per_share = half-spread in dollars), applied to every hedge trade, the
initial hedge, and the final unwind. cost=0 recovers the frictionless identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gamma_exit.pricing.black_scholes import bs_price
from gamma_exit.pricing.greeks import delta as bs_delta
from gamma_exit.pricing.greeks import gamma as bs_gamma


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma_real: float,
    t_years: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Exact GBM discretization: shape (n_paths, n_steps + 1), paths[:, 0] = s0.

    Exact in distribution at the grid points, so refining n_steps changes only
    the hedging frequency, not the quality of the price process itself.
    """
    rng = np.random.default_rng(seed)
    dt = t_years / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_paths)
    return paths


@dataclass(frozen=True)
class HedgeResult:
    """Per-path outputs of a synthetic delta-hedged replay."""

    pnl: np.ndarray  # (n_paths,) final portfolio value X_T (starts at 0)
    identity_integral: np.ndarray  # (n_paths,) closed-form pathwise integral
    pnl_path: np.ndarray  # (n_paths, n_steps + 1) X_t through time
    trading_cost: np.ndarray  # (n_paths,) total dollars paid in costs


def delta_hedge_synthetic(
    paths: np.ndarray,
    k: float,
    t_years: float,
    r: float,
    sigma_iv: float,
    sigma_real: float,
    kind: str = "call",
    q: float = 0.0,
    cost_per_share: float = 0.0,
) -> HedgeResult:
    """Replay a long-option delta hedge along simulated paths.

    The option is marked at BS(sigma_iv) throughout and settles to payoff at T.
    Rehedging happens at every grid point of `paths`; to study frequency,
    simulate paths with different n_steps.
    """
    n_paths, n_grid = paths.shape
    n_steps = n_grid - 1
    dt = t_years / n_steps
    times = np.linspace(0.0, t_years, n_grid)
    ttm = t_years - times  # time to maturity at each grid point

    # Option marks and deltas on the whole grid (bs_price handles ttm == 0).
    v = np.empty_like(paths)
    dlt = np.empty_like(paths)
    for j in range(n_grid):
        v[:, j] = bs_price(paths[:, j], k, ttm[j], r, sigma_iv, q, kind)
        dlt[:, j] = bs_delta(paths[:, j], k, ttm[j], r, sigma_iv, q, kind)

    growth = np.exp(r * dt)

    # t=0: pay V0 for the option, receive the short-sale proceeds -shares*S0.
    # Portfolio value X0 = V0 + shares*S0 + cash = 0 (minus initial trading cost).
    shares = -dlt[:, 0]
    cash = -v[:, 0] - shares * paths[:, 0]
    cost = np.abs(shares) * cost_per_share
    cash -= cost

    pnl_path = np.zeros((n_paths, n_grid))
    pnl_path[:, 0] = -cost

    for j in range(1, n_grid):
        cash *= growth
        s_j = paths[:, j]
        # mark portfolio before rebalancing (same value after: trades are self-financing)
        pnl_path[:, j] = v[:, j] + shares * s_j + cash
        if j < n_steps:
            new_shares = -dlt[:, j]
            trade = new_shares - shares
            cash -= trade * s_j
            step_cost = np.abs(trade) * cost_per_share
            cash -= step_cost
            cost += step_cost
            shares = new_shares
            pnl_path[:, j] -= step_cost
        else:
            # final unwind of the share hedge at S_T
            step_cost = np.abs(shares) * cost_per_share
            cost += step_cost
            pnl_path[:, j] -= step_cost

    pnl = pnl_path[:, -1]

    identity = gamma_pnl_identity_integral(
        paths, k, t_years, r, sigma_iv, sigma_real, kind=kind, q=q
    )
    return HedgeResult(
        pnl=pnl, identity_integral=identity, pnl_path=pnl_path, trading_cost=cost
    )


def gamma_pnl_identity_integral(
    paths: np.ndarray,
    k: float,
    t_years: float,
    r: float,
    sigma_iv: float,
    sigma_real: float,
    kind: str = "call",
    q: float = 0.0,
) -> np.ndarray:
    """Pathwise integral  X_T = int_0^T e^{r(T-u)} 0.5 Gamma_iv S^2 (sr^2 - siv^2) du.

    Left-endpoint Riemann sum on the path grid. Gamma is evaluated at the
    hedge vol sigma_iv (that is what the identity requires -- NOT realized vol).
    """
    n_grid = paths.shape[1]
    n_steps = n_grid - 1
    dt = t_years / n_steps
    times = np.linspace(0.0, t_years, n_grid)
    ttm = t_years - times

    vol_gap = sigma_real**2 - sigma_iv**2
    total = np.zeros(paths.shape[0])
    for j in range(n_steps):  # left endpoints only; Gamma(T) excluded
        g = bs_gamma(paths[:, j], k, ttm[j], r, sigma_iv, q, kind)
        total += np.exp(r * (t_years - times[j])) * 0.5 * g * paths[:, j] ** 2 * vol_gap * dt
    return total
