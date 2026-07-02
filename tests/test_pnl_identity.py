"""Milestone 1 gate: the synthetic delta-hedge engine must reconcile with the
pathwise gamma-vega identity

    X_T = int_0^T e^{r(T-u)} 1/2 Gamma_iv S_u^2 (sigma_real^2 - sigma_iv^2) du

as rehedge frequency rises, and must show NO drift (mu) bias -- the paper's
-(mu - r) S^2 Gamma term must not appear in a self-financing account.
"""

from functools import lru_cache

import numpy as np
import pytest

from gamma_exit.pnl.engine import delta_hedge_synthetic, simulate_gbm_paths

S0 = 100.0
K = 100.0
T = 1.0
R = 0.02
N_PATHS = 3000
SEED = 20260702


@lru_cache(maxsize=32)
def run(mu, sigma_real, sigma_iv, n_steps, kind="call", cost=0.0):
    paths = simulate_gbm_paths(S0, mu, sigma_real, T, n_steps, N_PATHS, SEED)
    return delta_hedge_synthetic(
        paths, K, T, R, sigma_iv, sigma_real, kind=kind, cost_per_share=cost
    )


class TestConvergenceToIdentity:
    """Long vol position: sigma_real > sigma_iv, drift well above r."""

    MU, SR, SIV = 0.08, 0.30, 0.20
    FREQS = (12, 52, 252, 1008)

    def test_pathwise_rms_error_shrinks_like_sqrt_dt(self):
        rms = []
        for n in self.FREQS:
            res = run(self.MU, self.SR, self.SIV, n)
            rms.append(float(np.sqrt(np.mean((res.pnl - res.identity_integral) ** 2))))
        # strictly decreasing, and 12 -> 1008 steps should shrink RMS by
        # ~sqrt(84) ~ 9x; require at least 4x to be robust to MC noise.
        assert all(a > b for a, b in zip(rms, rms[1:])), rms
        assert rms[0] / rms[-1] > 4.0, rms

    def test_mean_pnl_matches_mean_identity_at_high_frequency(self):
        res = run(self.MU, self.SR, self.SIV, self.FREQS[-1])
        resid = res.pnl - res.identity_integral
        se = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        assert abs(float(resid.mean())) < max(4 * se, 0.01)
        # sanity: positive gamma + realized > implied => positive mean P&L
        assert res.pnl.mean() > 0

    def test_holds_for_puts_too(self):
        res = run(self.MU, self.SR, self.SIV, 252, kind="put")
        resid = res.pnl - res.identity_integral
        se = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        assert abs(float(resid.mean())) < max(4 * se, 0.02)


class TestFairlyPricedVol:
    """sigma_real == sigma_iv: identity is exactly zero; discrete-hedge P&L is
    pure zero-mean noise regardless of drift."""

    def test_identity_integral_is_zero(self):
        res = run(0.08, 0.25, 0.25, 252)
        assert np.allclose(res.identity_integral, 0.0)

    def test_mean_pnl_is_zero_within_mc_error(self):
        res = run(0.08, 0.25, 0.25, 252)
        se = float(res.pnl.std(ddof=1) / np.sqrt(len(res.pnl)))
        assert abs(float(res.pnl.mean())) < 4 * se


class TestNoDriftBias:
    """Empirical check on the paper's -(mu - r) S^2 Gamma drift term.

    If that term were part of hedged P&L, running mu = r + 0.15 with zero vol
    gap would depress mean P&L by roughly (mu - r) * E[int S^2 Gamma dt] --
    tens of dollars here. A self-financing account shows no such bias.
    """

    SR = SIV = 0.25

    def test_high_drift_produces_no_pnl_bias(self):
        from gamma_exit.pricing.greeks import gamma as bs_gamma

        n_steps = 252
        mu_hi = R + 0.15
        res = run(mu_hi, self.SR, self.SIV, n_steps)

        se = float(res.pnl.std(ddof=1) / np.sqrt(len(res.pnl)))
        mean_pnl = float(res.pnl.mean())

        # magnitude the paper's term would predict on these same paths
        paths = simulate_gbm_paths(S0, mu_hi, self.SR, T, n_steps, N_PATHS, SEED)
        dt = T / n_steps
        times = np.linspace(0.0, T, n_steps + 1)
        acc = np.zeros(paths.shape[0])
        for j in range(n_steps):
            g = bs_gamma(paths[:, j], K, T - times[j], R, self.SIV)
            acc += (mu_hi - R) * paths[:, j] ** 2 * g * dt
        paper_bias = float(acc.mean())

        assert paper_bias > 10.0  # the claimed effect would be large...
        assert abs(mean_pnl) < 4 * se  # ...but measured P&L is unbiased
        assert abs(mean_pnl) < 0.1 * paper_bias

    def test_drift_does_not_move_mean_pnl(self):
        lo = run(R, self.SR, self.SIV, 252)
        hi = run(R + 0.15, self.SR, self.SIV, 252)
        pooled_se = float(
            np.sqrt(lo.pnl.var(ddof=1) / len(lo.pnl) + hi.pnl.var(ddof=1) / len(hi.pnl))
        )
        assert abs(float(hi.pnl.mean() - lo.pnl.mean())) < 5 * pooled_se


class TestAccountingBasics:
    def test_pnl_path_starts_at_zero_without_costs(self):
        res = run(0.08, 0.30, 0.20, 52)
        assert np.allclose(res.pnl_path[:, 0], 0.0)

    def test_terminal_mark_is_payoff(self):
        # with sigma_iv = sigma_real and r = 0 the mean pnl is ~0; more direct:
        # replay one deterministic-ish case and confirm the final option mark
        # used in pnl equals intrinsic (engine uses bs_price at ttm=0).
        from gamma_exit.pricing.black_scholes import bs_price

        assert bs_price(123.4, K, 0.0, R, 0.2) == pytest.approx(23.4)
