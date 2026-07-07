"""THE single time basis for the whole project.

Convention: TRADING-DAY YEARS, 252 trading days = 1.0
--------------------------------------------------------
Every `t`/`dt`/`T` and every annualized volatility in this codebase is in
trading-day years unless a name explicitly says otherwise. Rationale:
realized variance accrues when markets trade; weekends and holidays add
theta cost but no gamma income, which is precisely the trade-off under
study. The engine's identity term  1/2 * Gamma * S^2 * (sr^2 - siv^2) * dt
is only correct when the vols and dt share one basis.

Conversions happen at the DATA BOUNDARY, never downstream:
- Market option expiries are calendar dates -> `years_to_expiry` counts
  BUSINESS days to expiry and divides by 252. An IV solved from a market
  price using this T is a trading-basis vol, directly comparable to
  realized/forecast vols from vol/ (which annualize by 252).
- Business-day counting uses numpy's busday (weekends excluded). US
  exchange holidays (~9/yr) are NOT excluded: that overstates T by ~3.5%
  on average, a second-order effect on vol scale (~1.75%). Upgrade path:
  pass an `exchange_calendars` holiday list to `np.busday_count` here --
  this module is the only place that would change.

Accepted approximation -- interest on the trading clock:
Rates accrue on calendar time, but this project keeps r on the same
trading clock as everything else. I sized the error before accepting it:
for T_cal=60d vs T_trd=42d the exponent r*T differs by ~1.5% OF r --
sub-basis-point price impact at r <= 5% and holding horizons <= 1y.
Deliberate; revisit only for r >> 5% or multi-year options. (Two-clock
accounting would double the API surface for basis-point precision this
study does not need.)
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

TRADING_DAYS_PER_YEAR = 252
TRADING_DT = 1.0 / TRADING_DAYS_PER_YEAR


def trading_days_between(start: date, end: date) -> int:
    """Business days in [start, end) -- 0 if end <= start, never negative."""
    return int(np.busday_count(start, max(start, end)))


def years_to_expiry(expiry: date, asof: datetime | date) -> float:
    """Trading-day years from `asof` to expiry close.

    The expiry date itself counts as a trading day (an option expiring
    tomorrow has 1 business day of variance left; a 0-DTE option has the
    remaining fraction of today -- approximated here as a half day, since
    intraday clocks are out of scope for daily-bar research).
    """
    asof_date = asof.date() if isinstance(asof, datetime) else asof
    if expiry < asof_date:
        return 0.0
    if expiry == asof_date:
        return 0.5 * TRADING_DT
    # busdays in (asof_date, expiry]: count [asof, expiry), drop asof, add expiry
    days = (
        int(np.busday_count(asof_date, expiry))
        - int(np.is_busday(asof_date))
        + int(np.is_busday(expiry))
    )
    return max(days, 0) * TRADING_DT
