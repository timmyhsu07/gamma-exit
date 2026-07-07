"""Prototype end-to-end data path on the free yfinance feed.

Fetch a live chain snapshot + 1y of underlying bars, persist both through the
write-once cache, reload from disk, recompute IV from bid/ask mids on liquid
near-ATM contracts, and compare against the provider's IV column.

Run parameters (rate, quote filters, cache root, default ticker) come from
the experiment config. Recomputed IVs use trading-day-year T (see
gamma_exit.conventions), so they are directly comparable to
the realized/forecast vols printed alongside.

Run:  python -m gamma_exit.data.snapshot [^SPX] [--max-expiries 3]
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from gamma_exit.config import DEFAULT_CONFIG, load_config
from gamma_exit.conventions import years_to_expiry
from gamma_exit.data.cache import WriteOnceCache
from gamma_exit.data.providers.yfinance_provider import YFinanceProvider
from gamma_exit.pricing import implied_vol
from gamma_exit.vol.realized import close_to_close


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", nargs="?", default=None, help="default: first of config universe")
    ap.add_argument("--max-expiries", type=int, default=3)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache-root", default=None, help="default: config data.cache_root")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ticker_sym = args.ticker or cfg.data.universe[0]
    risk_free = cfg.rates.risk_free

    provider = YFinanceProvider()
    cache = WriteOnceCache(args.cache_root or cfg.data.cache_root)
    asof = datetime.now(timezone.utc)

    # --- pull + persist (write-once) ------------------------------------
    chain = provider.fetch_chain(ticker_sym, max_expiries=args.max_expiries)
    key_sym = ticker_sym.lstrip("^")  # cache keys are [A-Za-z0-9._/-] only
    chain_key = f"{key_sym}/{asof:%Y-%m-%dT%H%M%S}"
    cache.write(chain, "chains", chain_key)

    end = date.today()
    start = end - timedelta(days=365)
    bars_key = f"{key_sym}/{start}_{end}"
    if cache.exists("underlying", bars_key):
        # validate before reusing; a bad pull gets quarantined, never reused
        cached = cache.read("underlying", bars_key)
        if cached.empty:
            cache.quarantine("underlying", bars_key, "empty bars frame on reuse")
    if not cache.exists("underlying", bars_key):
        bars = provider.fetch_underlying(ticker_sym, start, end)
        cache.write(bars, "underlying", bars_key)

    # --- reload from disk: downstream code only ever sees the cache -----
    chain = cache.read("chains", chain_key)
    bars = cache.read("underlying", bars_key)

    spot = float(chain["underlying_price"].iloc[0])
    print(f"{ticker_sym} snapshot @ {asof:%Y-%m-%d %H:%M UTC}   spot={spot:.2f}")
    print(f"chain rows: {len(chain)}  expiries: {sorted(chain['expiry'].dt.date.unique())}")
    print(f"underlying bars: {len(bars)}  ({bars['date'].min():%Y-%m-%d} .. {bars['date'].max():%Y-%m-%d})")
    rv = close_to_close(bars.set_index("date")["close"].iloc[-63:])
    print(f"trailing 3m realized vol (close-to-close, EX-POST): {rv:.1%}")

    # --- liquid near-ATM subset; recompute IV from mids ------------------
    lo, hi = cfg.quotes.moneyness_band
    near_atm = chain[chain["strike"].between(lo * spot, hi * spot)].copy()
    liquid = near_atm[
        near_atm["mid"].notna()
        & (
            (near_atm["volume"].fillna(0) >= cfg.quotes.min_volume)
            | (near_atm["open_interest"].fillna(0) >= cfg.quotes.min_open_interest)
        )
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
        liquid = near_atm[
            near_atm["last"].notna() & (near_atm["volume"].fillna(0) >= cfg.quotes.min_volume)
        ].copy()
        price_col = "last"

    liquid["t_years"] = [years_to_expiry(e.date(), asof) for e in liquid["expiry"]]
    liquid = liquid[liquid["t_years"] > 0]
    liquid["iv_mid"] = [
        implied_vol(m, spot, k, t, risk_free, cfg.rates.dividend_yield, kind)
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
