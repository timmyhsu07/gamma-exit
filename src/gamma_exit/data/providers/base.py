"""Provider ABC. Implementations normalize raw payloads into the canonical
schema at the boundary and never leak provider-specific columns downstream."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class Provider(ABC):
    name: str

    @abstractmethod
    def fetch_chain(self, underlying: str, max_expiries: int | None = None) -> pd.DataFrame:
        """Current (or historical, where supported) option chain snapshot as a
        canonical options frame (schema.OPTION_COLUMNS)."""

    @abstractmethod
    def fetch_underlying(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily unadjusted OHLCV + dividends as a canonical bars frame."""

    def fetch_rates(self, start: date, end: date) -> pd.DataFrame:
        """Risk-free curve. Prototype uses a constant from config; real curve
        sourcing (FRED / OptionMetrics zero curve) lands with Milestone 2."""
        raise NotImplementedError(f"{self.name} does not provide rates")
