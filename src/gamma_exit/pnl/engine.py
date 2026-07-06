"""Delta-hedged P&L: one self-financing accounting core, two thin adapters.

Eng-review decision 1A: the accounting loop below is the ONLY code in this
project that turns positions into P&L. Synthetic mode (Milestone 1) feeds it
Black-Scholes marks on simulated paths; replay mode (Milestone 3) feeds it
observed mid quotes on real paths. The Milestone 1 identity tests therefore
validate the exact loop that produces real-data results.

    SYNTHETIC adapter                REPLAY adapter (M3)
    marks = BS(sigma_iv)             marks = observed mids
    deltas = BS-delta(sigma_iv)      deltas = BS-delta(day's IV)
            \\                          /
             v                        v
        replay_hedged_position()  <- the validated core
             |
             v
        HedgeResult (pnl_path, pnl, trading_cost)

The identity being tested (Milestone 1 gate, tests/test_pnl_identity.py)
------------------------------------------------------------------------
With the true path following dS = mu S dt + sigma_real S dW and the hedge run
at sigma_iv, continuous-time self-financing accounting gives, PATHWISE:

    X_T = integral_0^T e^{r (T-u)} * 1/2 * Gamma_iv(u, S_u) * S_u^2
              * (sigma_real^2 - sigma_iv^2) du

The drift mu does NOT appear; the source paper's -(mu - r) S^2 Gamma term is
an artifact of non-self-financing accounting, and tests assert its absence.
With a continuous dividend yield q the same identity holds provided the cash
account receives the dividend flow on the share position (the core does).

All times are TRADING-DAY YEARS (see gamma_exit.conventions).
Costs are proportional per share traded (cost_per_share = half-spread in
dollars), charged on the initial hedge, every rebalance, and the final unwind.
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
    """Per-path outputs of a delta-hedged replay."""

    pnl: np.ndarray  # (n_paths,) final portfolio value X_T (starts at 0)
    pnl_path: np.ndarray  # (n_paths, n_grid) X_t through time
    trading_cost: np.ndarray  # (n_paths,) total dollars paid in costs
    # money-market balance carried OUT of each grid point (post accrual,
    # dividends, and rebalance); attribution derives interest from it so the
    # cash recursion lives in exactly one place
    cash_path: np.ndarray  # (n_paths, n_grid)


def replay_hedged_position(
    times: np.ndarray,
    underlying: np.ndarray,
    option_marks: np.ndarray,
    hedge_ratios: np.ndarray,
    r: float,
    q: float = 0.0,
    cost_per_share: float = 0.0,
) -> HedgeResult:
    """THE self-financing accounting core (mark-agnostic; see module docstring).

    Inputs (n_paths, n_grid), times (n_grid,) in trading-day years:
    - option_marks[:, j]  value of the long option at t_j; the FINAL column
      must be the settlement/exit value (payoff at expiry, exit mid on close).
    - hedge_ratios[:, j]  the delta to be short over (t_j, t_{j+1}]; the final
      column is unused (the position unwinds at the last grid point).

    Accounting per step (t_{j-1} -> t_j], vectorized across paths:

        cash *= e^{r dt}                       # money-market accrual
        cash += shares * S_{j-1} * (e^{q dt}-1)  # dividend flow on shares
        X_j   = mark_j + shares * S_j + cash   # portfolio value at t_j
        rebalance: shares -> -hedge_ratios[:, j], cash absorbs the trade,
                   costs charged per share traded (final step: full unwind)

    X_0 = 0 minus the initial hedge cost; every later X_j is realizable value.
    """
    n_paths, n_grid = underlying.shape
    if option_marks.shape != underlying.shape or hedge_ratios.shape != underlying.shape:
        raise ValueError("underlying, option_marks, hedge_ratios must share one shape")
    if times.shape != (n_grid,):
        raise ValueError(f"times must have shape ({n_grid},), got {times.shape}")

    dts = np.diff(times)
    if (dts <= 0).any():
        raise ValueError("times must be strictly increasing")

    # t=0: pay mark_0 for the option, receive the short-sale proceeds.
    shares = -hedge_ratios[:, 0]
    cash = -option_marks[:, 0] - shares * underlying[:, 0]
    cost = np.abs(shares) * cost_per_share
    cash -= cost

    pnl_path = np.zeros((n_paths, n_grid))
    pnl_path[:, 0] = -cost
    cash_path = np.zeros((n_paths, n_grid))
    cash_path[:, 0] = cash

    for j in range(1, n_grid):
        dt = dts[j - 1]
        cash *= np.exp(r * dt)
        if q != 0.0:
            cash += shares * underlying[:, j - 1] * np.expm1(q * dt)
        s_j = underlying[:, j]
        pnl_path[:, j] = option_marks[:, j] + shares * s_j + cash
        if j < n_grid - 1:
            new_shares = -hedge_ratios[:, j]
            trade = new_shares - shares
            cash -= trade * s_j
            step_cost = np.abs(trade) * cost_per_share
            cash -= step_cost
            cost += step_cost
            shares = new_shares
            pnl_path[:, j] -= step_cost
        else:
            step_cost = np.abs(shares) * cost_per_share
            cost += step_cost
            pnl_path[:, j] -= step_cost
        cash_path[:, j] = cash

    return HedgeResult(
        pnl=pnl_path[:, -1], pnl_path=pnl_path, trading_cost=cost, cash_path=cash_path
    )


def delta_hedge_synthetic(
    paths: np.ndarray,
    k: float,
    t_years: float,
    r: float,
    sigma_iv: float,
    kind: str = "call",
    q: float = 0.0,
    cost_per_share: float = 0.0,
) -> HedgeResult:
    """Synthetic adapter: mark and hedge at BS(sigma_iv) along simulated paths.

    Rehedging happens at every grid point of `paths`; to study frequency,
    simulate paths with different n_steps. Compare the result against
    `gamma_pnl_identity_integral` (the closed-form reference) in tests.
    """
    n_grid = paths.shape[1]
    times = np.linspace(0.0, t_years, n_grid)
    ttm = t_years - times

    marks = np.empty_like(paths)
    deltas = np.empty_like(paths)
    for j in range(n_grid):
        marks[:, j] = bs_price(paths[:, j], k, ttm[j], r, sigma_iv, q, kind)
        deltas[:, j] = bs_delta(paths[:, j], k, ttm[j], r, sigma_iv, q, kind)

    return replay_hedged_position(
        times, paths, marks, deltas, r, q=q, cost_per_share=cost_per_share
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
    """Pathwise reference  X_T = int_0^T e^{r(T-u)} 0.5 Gamma_iv S^2 (sr^2 - siv^2) du.

    Left-endpoint Riemann sum on the path grid. Gamma is evaluated at the
    hedge vol sigma_iv (that is what the identity requires -- NOT realized
    vol). This is the ex-post REFERENCE the engine is validated against; it
    is not part of the trading account.
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
