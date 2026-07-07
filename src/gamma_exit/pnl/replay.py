"""Milestone 3: replay a delta-hedged position through OBSERVED option quotes.

This is the real-data adapter for the validated accounting core:

    cache (canonical chains)          synthetic engine (M1)
            |                                 |
    load_option_history()                     |
            v                                 v
    quotes frame ->  replay_position()  -> replay_hedged_position()  <- core
                          |
                          v
              ReplayResult.daily  (gamma / theta / vega / carry / cost / net)

Marking policy (hard rules):
- ENTRY and any pre-expiry EXIT happen at observed bid/ask mids or not at all
  (mid-or-drop). Option spread cost = option_spread_frac * observed half-spread
  on each side; exit AT expiry is a settlement, no spread.
- Days with no tradable mid are marked to model (BS at the last solved IV,
  flagged `stale`) so the position can be carried, but nothing trades on a
  stale mark and vega P&L is zero on those days by construction (IV ffill).
- The hedge trades the UNDERLYING (liquid regardless of option quotes), so
  the position re-hedges every day at the model delta.

Attribution: model-based decomposition with Greeks at the previous close.
`residual` is the exact plug making the rows sum to the true accounting net
(higher-order terms + quote noise); the equivalence test in
tests/test_replay.py pins it near zero on smooth synthetic chains, and the
identity net == sum(components) holds by construction on any data.

All times are trading-day years (see conventions.py). Prices are per-share;
scale by 100x contracts in the caller. Weekend/holiday snapshot rows are
dropped by the loader (a quote outside a trading session cannot be part of a
daily replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from gamma_exit.conventions import (
    TRADING_DAYS_PER_YEAR,
    trading_days_between,
    years_to_expiry,
)
from gamma_exit.data.cache import WriteOnceCache
from gamma_exit.pnl.engine import replay_hedged_position
from gamma_exit.pricing.black_scholes import bs_price
from gamma_exit.pricing.greeks import delta as bs_delta
from gamma_exit.pricing.greeks import gamma as bs_gamma
from gamma_exit.pricing.greeks import theta as bs_theta
from gamma_exit.pricing.greeks import vega as bs_vega
from gamma_exit.pricing.implied_vol import implied_vol
from gamma_exit.vol.realized import close_to_close

QUOTE_COLUMNS = ["date", "spot", "bid", "ask", "mid", "volume", "open_interest"]


@dataclass(frozen=True)
class PositionSpec:
    """One long-option position to replay (selection happens upstream, M5)."""

    underlying: str
    expiry: date
    strike: float
    kind: Literal["call", "put"]
    entry_date: date
    exit_date: date | None = None  # None = hold to expiry (settlement)

    @property
    def effective_exit(self) -> date:
        return self.exit_date if self.exit_date is not None else self.expiry


@dataclass(frozen=True)
class ReplayResult:
    daily: pd.DataFrame  # one row per trading day: attribution + cum_net
    pnl: float  # final X_T including all costs
    trading_cost: float  # total dollars paid (shares + option spread)
    summary: dict


def load_option_history(cache: WriteOnceCache, spec: PositionSpec) -> pd.DataFrame:
    """Per-trading-day quote history for one contract from cached chain pulls.

    Snapshots are deduped to the LAST pull per US-Eastern trading date, and
    rows outside business days are dropped (a weekend pull cannot be a daily
    mark). Returns the QUOTE_COLUMNS frame `replay_position` consumes.
    """
    sql = f"""
        select "asof", bid, ask, mid, volume, open_interest, underlying_price
        from read_parquet('{{root}}/chains/**/*.parquet')
        where underlying = '{spec.underlying}'
          and kind = '{spec.kind}'
          and strike = {float(spec.strike)!r}
          and cast(expiry as date) = DATE '{spec.expiry.isoformat()}'
        order by "asof"
    """
    df = cache.query(sql)
    if df.empty:
        raise ValueError(
            f"no cached quotes for {spec.underlying} {spec.expiry} "
            f"{spec.strike} {spec.kind}"
        )
    eastern = pd.to_datetime(df["asof"], utc=True).dt.tz_convert("America/New_York")
    df = df.assign(date=eastern.dt.date)
    df = df[np.is_busday(df["date"].tolist())]
    per_day = df.sort_values("asof").groupby("date", as_index=False).last()
    per_day = per_day.rename(columns={"underlying_price": "spot"})
    return per_day[QUOTE_COLUMNS].sort_values("date").reset_index(drop=True)


def _half_spread(row: pd.Series, what: str) -> float:
    bid, ask = float(row["bid"]), float(row["ask"])
    if not (np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask):
        raise ValueError(f"no two-sided quote at {what} ({row['date']}): bid={bid} ask={ask}")
    return 0.5 * (ask - bid)


def replay_position(
    quotes: pd.DataFrame,
    spec: PositionSpec,
    r: float,
    q: float = 0.0,
    share_cost_per_share: float = 0.0,
    option_spread_frac: float = 0.0,
) -> ReplayResult:
    """Replay one long option, delta-hedged daily at the close, through
    observed quotes. See module docstring for marking/cost policy."""
    exit_date = spec.effective_exit
    if exit_date > spec.expiry:
        raise ValueError(f"exit {exit_date} is after expiry {spec.expiry}")

    df = quotes.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= spec.entry_date) & (df["date"] <= exit_date)].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"need >= 2 quote days in [{spec.entry_date}, {exit_date}]")
    if df["date"].iloc[0] != spec.entry_date:
        raise ValueError(f"no quote row on entry date {spec.entry_date}")
    if df["date"].iloc[-1] != exit_date:
        raise ValueError(f"no quote row on exit date {exit_date}")

    n = len(df)
    dates = df["date"].to_numpy()
    spot = df["spot"].to_numpy(dtype=float)
    mid = df["mid"].to_numpy(dtype=float)
    if not np.isfinite(spot).all():
        raise ValueError("spot must be present on every replay day")

    settles_at_expiry = exit_date == spec.expiry
    times = np.array(
        [trading_days_between(spec.entry_date, d) / TRADING_DAYS_PER_YEAR for d in dates]
    )
    tte = np.array([years_to_expiry(spec.expiry, d) for d in dates])

    # --- mark-or-model: IV from mids where tradable, ffilled elsewhere -----
    quote_ok = np.isfinite(mid)
    if not quote_ok[0]:
        raise ValueError(f"entry {spec.entry_date}: no tradable mid (mid-or-drop)")
    if not settles_at_expiry and not quote_ok[-1]:
        raise ValueError(f"exit {exit_date}: no tradable mid (mid-or-drop)")

    iv_raw = np.full(n, np.nan)
    solve = quote_ok.copy()
    if settles_at_expiry:
        solve[-1] = False  # settlement row needs no vol
    if solve.any():
        iv_raw[solve] = implied_vol(
            mid[solve], spot[solve], spec.strike, tte[solve], r, q, spec.kind
        )
    if not np.isfinite(iv_raw[0]):
        raise ValueError(
            f"entry {spec.entry_date}: mid {mid[0]} unsolvable for IV "
            "(below intrinsic or degenerate) -- position not enterable"
        )
    iv = pd.Series(iv_raw).ffill().to_numpy()
    stale = ~quote_ok | ~np.isfinite(iv_raw)  # marked/valued without a fresh quote
    if settles_at_expiry:
        stale[-1] = False  # settlement is exact, not stale

    marks = np.where(quote_ok, mid, bs_price(spot, spec.strike, tte, r, iv, q, spec.kind))
    if settles_at_expiry:
        marks[-1] = bs_price(spot[-1], spec.strike, 0.0, r, iv[-1], q, spec.kind)  # intrinsic

    deltas = bs_delta(spot, spec.strike, tte, r, iv, q, spec.kind)

    # --- the validated core does ALL the accounting ------------------------
    res = replay_hedged_position(
        times,
        spot[None, :],
        marks[None, :],
        deltas[None, :],
        r,
        q=q,
        cost_per_share=share_cost_per_share,
    )

    # option spread costs on top (entry always; exit only if sold pre-expiry)
    entry_cost = option_spread_frac * _half_spread(df.iloc[0], "entry")
    exit_cost = (
        0.0 if settles_at_expiry else option_spread_frac * _half_spread(df.iloc[-1], "exit")
    )
    pnl_path = res.pnl_path[0] - entry_cost
    pnl_path[-1] -= exit_cost
    trading_cost = float(res.trading_cost[0]) + entry_cost + exit_cost

    # --- daily attribution (Greeks at previous close; residual = exact plug)
    cash = res.cash_path[0]
    d_spot = np.diff(spot)
    dts = np.diff(times)
    g_prev = bs_gamma(spot[:-1], spec.strike, tte[:-1], r, iv[:-1], q, spec.kind)
    th_prev = bs_theta(spot[:-1], spec.strike, tte[:-1], r, iv[:-1], q, spec.kind)
    ve_prev = bs_vega(spot[:-1], spec.strike, tte[:-1], r, iv[:-1], q, spec.kind)

    gamma_pnl = np.zeros(n)
    theta_pnl = np.zeros(n)
    vega_pnl = np.zeros(n)
    carry = np.zeros(n)
    gamma_pnl[1:] = 0.5 * g_prev * d_spot**2
    theta_pnl[1:] = th_prev * dts
    vega_pnl[1:] = ve_prev * np.diff(iv)
    carry[1:] = cash[:-1] * np.expm1(r * dts) - deltas[:-1] * spot[:-1] * np.expm1(q * dts)

    cost_day = np.zeros(n)
    cost_day[0] = np.abs(deltas[0]) * share_cost_per_share + entry_cost
    if n > 2:
        cost_day[1:-1] = np.abs(np.diff(deltas[:-1])) * share_cost_per_share
    cost_day[-1] = np.abs(deltas[-2]) * share_cost_per_share + exit_cost

    net = np.diff(pnl_path, prepend=0.0)
    residual = net - (gamma_pnl + theta_pnl + vega_pnl + carry - cost_day)

    # two-sided-quote diagnostics: where an exit could actually trade
    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    two_sided = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    half_spread = np.where(two_sided, 0.5 * (ask - bid), np.nan)
    tradable = quote_ok & two_sided
    if settles_at_expiry:
        tradable[-1] = True  # settlement needs no quote

    daily = pd.DataFrame(
        {
            "date": dates,
            "spot": spot,
            "mark": marks,
            "iv": iv,
            "stale": stale,
            "tradable": tradable,
            "half_spread": half_spread,
            "delta": deltas,
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "vega_pnl": vega_pnl,
            "carry": carry,
            "cost": cost_day,
            "residual": residual,
            "net": net,
            "cum_net": pnl_path,
        }
    )

    summary = {
        "underlying": spec.underlying,
        "kind": spec.kind,
        "strike": spec.strike,
        "expiry": spec.expiry,
        "entry_date": spec.entry_date,
        "exit_date": exit_date,
        "exit_kind": "expiry_settlement" if settles_at_expiry else "market_close",
        "n_days": n,
        "stale_days": int(stale.sum()),
        "entry_spot": float(spot[0]),
        "entry_iv": float(iv[0]),
        "entry_mark": float(marks[0]),
        "realized_vol_window": close_to_close(pd.Series(spot)),  # EX-POST only
        "pnl": float(pnl_path[-1]),
        "trading_cost": trading_cost,
        # recorded so exit_values() reprices early exits at the SAME costs
        "share_cost_per_share": share_cost_per_share,
        "option_spread_frac": option_spread_frac,
    }
    return ReplayResult(
        daily=daily, pnl=float(pnl_path[-1]), trading_cost=trading_cost, summary=summary
    )


def exit_values(result: ReplayResult) -> tuple[np.ndarray, np.ndarray]:
    """Realizable P&L of exiting at each day of a HOLD-TO-EXPIRY replay.

    Returns (values, executable): values[e] is the cost-adjusted portfolio
    value if the position were closed at day e's close -- sell the option at
    mid minus the spread charge, unwind the hedge shares held into day e --
    and executable[e] says whether that exit could actually trade (two-sided
    quote; day 0 never, the final settlement day always).

    Derivation from the replayed path: cum_net[e] already reflects day e's
    REBALANCE cost, which an exiting trader would not pay; add it back, then
    charge the full hedge unwind and the option exit spread. Verified in
    tests against an explicit replay with exit_date=e (exact equality).
    """
    d = result.daily
    n = len(d)
    frac = result.summary["option_spread_frac"]
    share_cost = result.summary["share_cost_per_share"]

    cum = d["cum_net"].to_numpy()
    cost = d["cost"].to_numpy()
    delta = d["delta"].to_numpy()
    hs = d["half_spread"].to_numpy()
    tradable = d["tradable"].to_numpy(dtype=bool)

    values = np.full(n, np.nan)
    executable = tradable.copy()
    executable[0] = False  # cannot exit at the entry close
    interior = np.arange(1, n - 1)
    values[interior] = (
        cum[interior]
        + cost[interior]
        - np.abs(delta[interior - 1]) * share_cost
        - frac * hs[interior]
    )
    values[-1] = result.pnl if executable[-1] else np.nan
    return values, executable
