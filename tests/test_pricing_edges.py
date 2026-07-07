"""Degenerate-branch coverage for the pricing layer: the grid
replay hits t=0 and callers may pass sigma=0; those limits must be exact, and
bad `kind` strings must fail loudly."""

import numpy as np
import pytest

from gamma_exit.pricing.black_scholes import bs_price
from gamma_exit.pricing.greeks import delta, gamma, theta, vega


class TestSigmaZeroLimit:
    def test_call_is_discounted_forward_intrinsic(self):
        s, k, t, r = 110.0, 100.0, 0.5, 0.04
        want = s - k * np.exp(-r * t)
        assert bs_price(s, k, t, r, 0.0) == pytest.approx(want)
        # OTM-forward call is worthless without volatility
        assert bs_price(80.0, 100.0, 0.5, r, 0.0) == 0.0

    def test_put_side(self):
        s, k, t, r = 80.0, 100.0, 0.5, 0.04
        want = k * np.exp(-r * t) - s
        assert bs_price(s, k, t, r, 0.0, kind="put") == pytest.approx(want)

    def test_with_dividend_yield(self):
        s, k, t, r, q = 110.0, 100.0, 1.0, 0.04, 0.03
        want = s * np.exp(-q * t) - k * np.exp(-r * t)
        assert bs_price(s, k, t, r, 0.0, q) == pytest.approx(want)

    def test_sigma_zero_greeks_are_finite(self):
        for fn in (gamma, vega, theta):
            assert fn(100.0, 100.0, 0.5, 0.02, 0.0) == 0.0


class TestExpiryLimit:
    def test_delta_is_intrinsic_step(self):
        assert delta(110.0, 100.0, 0.0, 0.02, 0.2) == 1.0
        assert delta(90.0, 100.0, 0.0, 0.02, 0.2) == 0.0
        assert delta(90.0, 100.0, 0.0, 0.02, 0.2, kind="put") == -1.0
        assert delta(110.0, 100.0, 0.0, 0.02, 0.2, kind="put") == 0.0

    def test_grid_mixing_live_and_expired_stays_finite(self):
        t = np.array([0.5, 0.0, 1e-15])
        out = bs_price(100.0, 100.0, t, 0.02, 0.2)
        assert np.isfinite(out).all()
        assert out[1] == 0.0  # ATM at expiry = intrinsic 0


class TestKindValidation:
    def test_bad_kind_raises_for_every_function(self):
        for fn in (bs_price, delta, gamma, vega, theta):
            with pytest.raises(ValueError, match="call.*put|put.*call"):
                fn(100.0, 100.0, 0.5, 0.02, 0.2, 0.0, "cal")
