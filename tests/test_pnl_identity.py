"""Gate 1: the self-financing accounting core must reconcile with the
pathwise gamma-vega identity

    X_T = int_0^T e^{r(T-u)} 1/2 Gamma_iv S_u^2 (sigma_real^2 - sigma_iv^2) du

as rehedge frequency rises, and must show NO drift (mu) bias -- the paper's
-(mu - r) S^2 Gamma term must not appear in a self-financing account.

Post-1A refactor these tests exercise `replay_hedged_position` (the shared
core) through the synthetic adapter, so the validation covers the exact loop
replay mode will use on real data.
"""

from functools import lru_cache

import numpy as np
import pytest

from gamma_exit.pnl.engine import (
    delta_hedge_synthetic,
    gamma_pnl_identity_integral,
    simulate_gbm_paths,
)

S0 = 100.0
K = 100.0
T = 1.0
R = 0.02
N_PATHS = 3000
SEED = 20260702


@lru_cache(maxsize=32)
def run(mu, sigma_real, sigma_iv, n_steps, kind="call", cost=0.0, q=0.0):
    """Hedge replay + pathwise identity reference on the same paths."""
    paths = simulate_gbm_paths(S0, mu, sigma_real, T, n_steps, N_PATHS, SEED)
    res = delta_hedge_synthetic(
        paths, K, T, R, sigma_iv, kind=kind, q=q, cost_per_share=cost
    )
    identity = gamma_pnl_identity_integral(
        paths, K, T, R, sigma_iv, sigma_real, kind=kind, q=q
    )
    return res, identity


class TestConvergenceToIdentity:
    """Long vol position: sigma_real > sigma_iv, drift well above r."""

    MU, SR, SIV = 0.08, 0.30, 0.20
    FREQS = (12, 52, 252, 1008)

    def test_pathwise_rms_error_shrinks_like_sqrt_dt(self):
        rms = []
        for n in self.FREQS:
            res, identity = run(self.MU, self.SR, self.SIV, n)
            rms.append(float(np.sqrt(np.mean((res.pnl - identity) ** 2))))
        # strictly decreasing, and 12 -> 1008 steps should shrink RMS by
        # ~sqrt(84) ~ 9x; require at least 4x to be robust to MC noise.
        assert all(a > b for a, b in zip(rms, rms[1:])), rms
        assert rms[0] / rms[-1] > 4.0, rms

    def test_mean_pnl_matches_mean_identity_at_high_frequency(self):
        res, identity = run(self.MU, self.SR, self.SIV, self.FREQS[-1])
        resid = res.pnl - identity
        se = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        assert abs(float(resid.mean())) < max(4 * se, 0.01)
        # sanity: positive gamma + realized > implied => positive mean P&L
        assert res.pnl.mean() > 0

    def test_holds_for_puts_too(self):
        res, identity = run(self.MU, self.SR, self.SIV, 252, kind="put")
        resid = res.pnl - identity
        se = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        assert abs(float(resid.mean())) < max(4 * se, 0.02)


class TestDividendYield:
    """The q parameter must reconcile, not just exist.

    With a continuous dividend yield the identity holds only if the core
    credits the dividend flow on the share position -- so this test fails
    loudly if that accounting is ever dropped.
    """

    MU, SR, SIV, Q = 0.06, 0.30, 0.20, 0.03

    def test_identity_holds_with_dividends(self):
        res, identity = run(self.MU, self.SR, self.SIV, 504, q=self.Q)
        resid = res.pnl - identity
        se = float(resid.std(ddof=1) / np.sqrt(len(resid)))
        assert abs(float(resid.mean())) < max(4 * se, 0.02)

    def test_dividend_flow_matters(self):
        """Same paths hedged with q in the Greeks but no dividend accounting
        would show a bias ~ q * E[int Delta S dt] -- assert our resid is far
        below that scale, proving the flow is actually credited."""
        res, identity = run(self.MU, self.SR, self.SIV, 504, q=self.Q)
        resid_mean = abs(float((res.pnl - identity).mean()))
        # rough scale of the missing-dividend bias: q * Delta * S * T ~ 0.03*0.6*100*1
        assert resid_mean < 0.1 * (self.Q * 0.6 * S0 * T)


class TestFairlyPricedVol:
    """sigma_real == sigma_iv: identity is exactly zero; discrete-hedge P&L is
    pure zero-mean noise regardless of drift."""

    def test_identity_integral_is_zero(self):
        _, identity = run(0.08, 0.25, 0.25, 252)
        assert np.allclose(identity, 0.0)

    def test_mean_pnl_is_zero_within_mc_error(self):
        res, _ = run(0.08, 0.25, 0.25, 252)
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
        res, _ = run(mu_hi, self.SR, self.SIV, n_steps)

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
        lo, _ = run(R, self.SR, self.SIV, 252)
        hi, _ = run(R + 0.15, self.SR, self.SIV, 252)
        pooled_se = float(
            np.sqrt(lo.pnl.var(ddof=1) / len(lo.pnl) + hi.pnl.var(ddof=1) / len(hi.pnl))
        )
        assert abs(float(hi.pnl.mean() - lo.pnl.mean())) < 5 * pooled_se


class TestGbmSimulator:
    """Statistical sanity of the path generator itself."""

    def test_log_return_moments(self):
        mu, sigma, t, n_steps, n_paths = 0.07, 0.22, 1.0, 252, 4000
        paths = simulate_gbm_paths(S0, mu, sigma, t, n_steps, n_paths, seed=11)
        log_rets = np.diff(np.log(paths), axis=1)
        dt = t / n_steps

        want_mean = (mu - 0.5 * sigma**2) * dt
        want_std = sigma * np.sqrt(dt)
        got_mean = float(log_rets.mean())
        got_std = float(log_rets.std(ddof=1))

        se_mean = want_std / np.sqrt(log_rets.size)
        assert abs(got_mean - want_mean) < 5 * se_mean
        assert abs(got_std - want_std) / want_std < 0.01

    def test_seed_reproducibility(self):
        a = simulate_gbm_paths(S0, 0.05, 0.2, 1.0, 10, 5, seed=42)
        b = simulate_gbm_paths(S0, 0.05, 0.2, 1.0, 10, 5, seed=42)
        assert np.array_equal(a, b)


class TestAccountingBasics:
    def test_pnl_path_starts_at_zero_without_costs(self):
        res, _ = run(0.08, 0.30, 0.20, 52)
        assert np.allclose(res.pnl_path[:, 0], 0.0)

    def test_terminal_mark_is_payoff(self):
        from gamma_exit.pricing.black_scholes import bs_price

        assert bs_price(123.4, K, 0.0, R, 0.2) == pytest.approx(23.4)

    def test_core_rejects_mismatched_shapes(self):
        from gamma_exit.pnl.engine import replay_hedged_position

        times = np.linspace(0.0, 1.0, 4)
        good = np.ones((2, 4))
        with pytest.raises(ValueError, match="share one shape"):
            replay_hedged_position(times, good, np.ones((2, 3)), good, R)
        with pytest.raises(ValueError, match="strictly increasing"):
            replay_hedged_position(times[::-1].copy(), good, good, good, R)
