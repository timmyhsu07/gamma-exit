"""Reconcile our analytic BS price and Greeks against QuantLib.

QuantLib is an independent implementation; agreement to tight tolerance on a
grid of moneyness / maturity / vol / rates is the Milestone 1 pricing gate.
"""

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from gamma_exit.pricing import bs_price, delta, gamma, theta, vega  # noqa: E402

GRID = [
    # s, k, t, r, sigma, q
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 100.0, 0.25, 0.05, 0.20, 0.00),
    (100.0, 100.0, 0.02, 0.05, 0.40, 0.00),
    (80.0, 100.0, 0.50, 0.03, 0.35, 0.01),
    (120.0, 100.0, 0.50, 0.03, 0.35, 0.01),
    (100.0, 100.0, 2.00, 0.00, 0.15, 0.02),
    (95.0, 100.0, 0.10, 0.07, 0.55, 0.00),
    (150.0, 100.0, 1.50, 0.02, 0.10, 0.03),
    (60.0, 100.0, 1.00, 0.05, 0.25, 0.00),
]


def _quantlib_option(s, k, t, r, sigma, q, kind):
    today = ql.Date(1, 7, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    expiry = today + ql.Period(int(round(t * 365)), ql.Days)

    spot = ql.QuoteHandle(ql.SimpleQuote(s))
    rf = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    div = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, dc))
    process = ql.BlackScholesMertonProcess(spot, div, rf, vol)

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, k
    )
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    # use QuantLib's own year fraction so both sides price the same T
    t_ql = dc.yearFraction(today, expiry)
    return option, t_ql


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("s,k,t,r,sigma,q", GRID)
def test_price_and_greeks_match_quantlib(s, k, t, r, sigma, q, kind):
    option, t_ql = _quantlib_option(s, k, t, r, sigma, q, kind)

    assert bs_price(s, k, t_ql, r, sigma, q, kind) == pytest.approx(
        option.NPV(), abs=1e-10
    )
    assert delta(s, k, t_ql, r, sigma, q, kind) == pytest.approx(option.delta(), abs=1e-10)
    assert gamma(s, k, t_ql, r, sigma, q, kind) == pytest.approx(option.gamma(), abs=1e-10)
    assert vega(s, k, t_ql, r, sigma, q, kind) == pytest.approx(option.vega(), abs=1e-8)
    assert theta(s, k, t_ql, r, sigma, q, kind) == pytest.approx(option.theta(), abs=1e-8)


def test_put_call_parity_on_grid():
    for s, k, t, r, sigma, q in GRID:
        c = bs_price(s, k, t, r, sigma, q, "call")
        p = bs_price(s, k, t, r, sigma, q, "put")
        parity = s * np.exp(-q * t) - k * np.exp(-r * t)
        assert c - p == pytest.approx(parity, abs=1e-10)


def test_expiry_returns_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put") == pytest.approx(10.0)
    assert gamma(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == 0.0
