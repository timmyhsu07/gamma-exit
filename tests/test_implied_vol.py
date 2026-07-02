"""Round-trip and edge-case behavior of the implied vol solver."""

import numpy as np
import pytest

from gamma_exit.pricing import bs_price, implied_vol


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.5])
@pytest.mark.parametrize("s,k,t", [(100, 100, 1.0), (80, 100, 0.5), (120, 100, 0.1)])
def test_round_trip(s, k, t, sigma, kind):
    from gamma_exit.pricing import vega

    price = bs_price(s, k, t, 0.03, sigma, 0.01, kind)
    iv = implied_vol(price, s, k, t, 0.03, 0.01, kind)
    # The solver guarantees the PRICE round-trips. The VOL itself is only
    # identified where vega is non-negligible; deep-ITM short-dated low-vol
    # quotes are vega-degenerate (why the data layer filters such quotes).
    assert bs_price(s, k, t, 0.03, iv, 0.01, kind) == pytest.approx(price, abs=1e-8)
    if vega(s, k, t, 0.03, sigma, 0.01, kind) > 1e-4:
        assert iv == pytest.approx(sigma, abs=1e-6)


def test_below_intrinsic_returns_nan():
    # call worth less than its lower no-arbitrage bound has no implied vol
    assert np.isnan(implied_vol(0.5, 120.0, 100.0, 0.5, 0.03, 0.0, "call"))


def test_expired_returns_nan():
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, 0.03, 0.0, "call"))


def test_vectorized_matches_scalar():
    prices = np.array([bs_price(100, 100, 1.0, 0.03, v) for v in (0.1, 0.3)])
    ivs = implied_vol(prices, 100.0, 100.0, 1.0, 0.03, 0.0, "call")
    assert ivs == pytest.approx([0.1, 0.3], abs=1e-7)
