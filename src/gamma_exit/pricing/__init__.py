"""Black-Scholes pricing, implied volatility, and Greeks.

All functions are vectorized over numpy arrays and validated against
QuantLib in tests/test_greeks_vs_quantlib.py.
"""

from gamma_exit.pricing.black_scholes import bs_price, d1_d2
from gamma_exit.pricing.greeks import delta, gamma, theta, vega
from gamma_exit.pricing.implied_vol import implied_vol

__all__ = ["bs_price", "d1_d2", "delta", "gamma", "theta", "vega", "implied_vol"]
