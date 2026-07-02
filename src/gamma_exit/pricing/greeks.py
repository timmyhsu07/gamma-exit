"""Analytic Black-Scholes-Merton Greeks.

Conventions (must match QuantLib's analytic European engine, which the tests
enforce):
- delta = dV/dS
- gamma = d2V/dS2
- theta = dV/dt with t = calendar time (NOT time-to-expiry), in value per YEAR.
  Long options normally have negative theta. Divide by 365 for a per-day theta.
- vega  = dV/dsigma, per 1.0 of vol (divide by 100 for per-vol-point).

At t <= 0 or sigma <= 0 the Greeks are returned as 0 (delta as the intrinsic
step function), which is the limit almost everywhere and keeps grid replays
finite; the ATM-at-expiry singularity is deliberately not represented.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from gamma_exit.pricing.black_scholes import _EPS_SIGMA, _EPS_T, _validate_kind, d1_d2


def _degenerate(t, sigma):
    t = np.asarray(t)
    sigma = np.asarray(sigma)
    return (t <= _EPS_T) | (sigma <= _EPS_SIGMA)


def delta(s, k, t, r, sigma, q=0.0, kind: str = "call"):
    phi = _validate_kind(kind)
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    val = phi * np.exp(-np.asarray(q) * np.asarray(t)) * norm.cdf(phi * d1)
    # degenerate limit: intrinsic step function
    s_, k_ = np.asarray(s), np.asarray(k)
    step = np.where(phi * (s_ - k_) > 0, phi, 0.0)
    out = np.where(_degenerate(t, sigma), step, val)
    return out if out.ndim else float(out)


def gamma(s, k, t, r, sigma, q=0.0, kind: str = "call"):
    _validate_kind(kind)  # gamma identical for calls and puts
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    s_, t_, sig_ = np.asarray(s, dtype=float), np.asarray(t), np.asarray(sigma)
    val = np.exp(-np.asarray(q) * t_) * norm.pdf(d1) / (s_ * sig_ * np.sqrt(t_))
    out = np.where(_degenerate(t, sigma), 0.0, val)
    return out if out.ndim else float(out)


def vega(s, k, t, r, sigma, q=0.0, kind: str = "call"):
    _validate_kind(kind)  # vega identical for calls and puts
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    s_, t_ = np.asarray(s, dtype=float), np.asarray(t)
    val = s_ * np.exp(-np.asarray(q) * t_) * norm.pdf(d1) * np.sqrt(t_)
    out = np.where(_degenerate(t, sigma), 0.0, val)
    return out if out.ndim else float(out)


def theta(s, k, t, r, sigma, q=0.0, kind: str = "call"):
    """Calendar theta dV/dt (per year); negative for most long options."""
    phi = _validate_kind(kind)
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    s_, k_, t_, r_, sig_, q_ = map(np.asarray, (s, k, t, r, sigma, q))
    s_ = s_.astype(float)
    decay = -s_ * np.exp(-q_ * t_) * norm.pdf(d1) * sig_ / (2.0 * np.sqrt(t_))
    carry = phi * (
        q_ * s_ * np.exp(-q_ * t_) * norm.cdf(phi * d1)
        - r_ * k_ * np.exp(-r_ * t_) * norm.cdf(phi * d2)
    )
    out = np.where(_degenerate(t, sigma), 0.0, decay + carry)
    return out if out.ndim else float(out)
