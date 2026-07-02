"""Prototype end-to-end data path on the free yfinance feed.

Fetch a live chain snapshot + 1y of underlying bars, persist both through the
write-once cache, reload from disk, recompute IV from bid/ask mids on liquid
near-ATM contracts, and compare against the provider's IV column.

Run:  python -m gamma_exit.data.snapshot SPY --max-expiries 3
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from gamma_exit.data.cache import WriteOnceCache
from gamma_exit.data.providers.yfinance_provider import YFinanceProvider
from gamma_exit.pricing import implied_vol
from gamma_exit.vol.realized import close_to_close

RISK_FREE_PROTOTYPE = 0.04  # constant stub; real curve arrives with Milestone 2
YEAR_SECONDS = 365.0 * 24 * 3600


def years_to_expiry(expiry: pd.Timestamp, asof: datetime) -> float:
    """Time to a 20:00 UTC (4pm ET, DST) expiry close, in calendar years."""
    expiry_dt = expiry.to_pydatetime().replace(hour=20, tzinfo=timezone.utc)
    return max((expiry_dt - asof).total_seconds(), 0.0) / YEAR_SECONDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--max-expiries", type=int, default=3)
    ap.add_argument("--cache-root", default="cache")
    args = ap.parse_args()

    provider = YFinanceProvider()
    cache = WriteOnceCache(args.cache_root)
    asof = datetime.now(timezone.utc)

    # --- pull + persist (write-once) ------------------------------------
    chain = provider.fetch_chain(args.ticker, max_expiries=args.max_expiries)
    chain_key = f"{args.ticker}/{asof:%Y-%m-%dT%H%M%S}"
    cache.write(chain, "chains", chain_key)

    end = date.today()
    start = end - timedelta(days=365)
    bars = provider.fetch_underlying(args.ticker, start, end)
    bars_key = f"{args.ticker}/{start}_{end}"
    if not cache.exists("underlying", bars_key):
        cache.write(bars, "underlying", bars_key)

    # --- reload from disk: downstream code only ever sees the cache -----
    chain = cache.read("chains", chain_key)
    bars = cache.read("underlying", bars_key)

    spot = float(chain["underlying_price"].iloc[0])
    print(f"{args.ticker} snapshot @ {asof:%Y-%m-%d %H:%M UTC}   spot={spot:.2f}")
    print(f"chain rows: {len(chain)}  expiries: {sorted(chain['expiry'].dt.date.unique())}")
    print(f"underlying bars: {len(bars)}  ({bars['date'].min():%Y-%m-%d} .. {bars['date'].max():%Y-%m-%d})")
    rv = close_to_close(bars.set_index("date")["close"].iloc[-63:])
    print(f"trailing 3m realized vol (close-to-close, EX-POST): {rv:.1%}")

    # --- liquid near-ATM subset; recompute IV from mids ------------------
    near_atm = chain[chain["strike"].between(0.93 * spot, 1.07 * spot)].copy()
    liquid = near_atm[
        near_atm["mid"].notna()
        & ((near_atm["volume"].fillna(0) >= 10) | (near_atm["open_interest"].fillna(0) >= 100))
    ].copy()
    price_col = "mid"
    if liquid.empty:
        # Market closed: Yahoo zeroes bid/ask after hours, so no tradable mid
        # exists in this snapshot. Fall back to last trade for the DEMO ONLY;
        # replay backtests never do this (mid-or-drop is the hard rule).
        print(
            "\nNOTE: no two-sided quotes in this snapshot (market closed?). "
            "Falling back to STALE last-trade prices for the IV demo."
        )
        liquid = near_atm[near_atm["last"].notna() & (near_atm["volume"].fillna(0) >= 10)].copy()
        price_col = "last"

    liquid["t_years"] = [years_to_expiry(e, asof) for e in liquid["expiry"]]
    liquid = liquid[liquid["t_years"] > 0]
    liquid["iv_mid"] = [
        implied_vol(m, spot, k, t, RISK_FREE_PROTOTYPE, 0.0, kind)
        for m, k, t, kind in zip(
            liquid[price_col], liquid["strike"], liquid["t_years"], liquid["kind"]
        )
    ]
    ok = liquid[np.isfinite(liquid["iv_mid"])]
    gap = (ok["iv_mid"] - ok["provider_iv"]).abs()

    print(f"\nliquid near-ATM contracts: {len(liquid)}  (IV solvable: {len(ok)})")
    print(f"our mid-IV vs provider IV  median |gap|: {gap.median():.4f}  90pct: {gap.quantile(0.9):.4f}")
    print("\nsample (closest to ATM per expiry/kind):")
    sample = (
        ok.assign(atm_dist=(ok["strike"] - spot).abs())
        .sort_values("atm_dist")
        .groupby([ok["expiry"].dt.date, "kind"])
        .head(1)
        .sort_values(["expiry", "kind"])
    )
    cols = ["expiry", "kind", "strike", "bid", "ask", "mid", "iv_mid", "provider_iv", "volume"]
    with pd.option_context("display.width", 120):
        print(sample[cols].to_string(index=False))

    print(f"\ncache -> {cache.path_for('chains', chain_key)}")


if __name__ == "__main__":
    main()
