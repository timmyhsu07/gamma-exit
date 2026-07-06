"""Canonical data schema shared by every provider.

Providers normalize into these shapes at the boundary; everything downstream
(engine, policies, analytics) sees only canonical frames, never provider
quirks. Prices are per-share (an option contract is 100x these numbers).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, field_validator


class OptionRecord(BaseModel):
    """One option quote snapshot."""

    asof: datetime  # when the snapshot was taken (UTC)
    provider: str
    underlying: str
    expiry: date
    strike: float
    kind: Literal["call", "put"]
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    provider_iv: Optional[float] = None  # provider-reported IV; treat as advisory
    underlying_price: Optional[float] = None

    @field_validator("asof")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("asof must be timezone-aware (UTC)")
        return v

    @property
    def mid(self) -> Optional[float]:
        """Bid/ask mid; None when either side is missing or non-positive.

        Mid is the tradable mark this project uses -- last trades go stale on
        illiquid contracts and must not silently substitute for it.
        """
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return None
        return 0.5 * (self.bid + self.ask)


class UnderlyingBar(BaseModel):
    """One daily OHLCV bar of the underlying, unadjusted, with cash dividend."""

    provider: str
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float] = None
    volume: Optional[int] = None
    dividend: float = 0.0


OPTION_COLUMNS = list(OptionRecord.model_fields) + ["mid"]
BAR_COLUMNS = list(UnderlyingBar.model_fields)


def options_to_frame(records: list[OptionRecord]) -> pd.DataFrame:
    rows = [{**r.model_dump(), "mid": r.mid} for r in records]
    df = pd.DataFrame(rows, columns=OPTION_COLUMNS)
    # fail loudly at the boundary: malformed expiries must not flow downstream;
    # pin ns resolution so parquet round-trips compare equal
    df["expiry"] = pd.to_datetime(df["expiry"]).astype("datetime64[ns]")
    return df


def bars_to_frame(records: list[UnderlyingBar]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in records], columns=BAR_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df
