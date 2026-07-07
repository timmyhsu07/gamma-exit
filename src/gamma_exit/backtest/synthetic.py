"""Synthetic position source: GBM worlds with a known vol gap.

Generates what a perfect vendor would have recorded -- canonical quote
frames (mid = BS(sigma_iv) marks, bid/ask around them) plus pre-entry
underlying history for the causal forecast -- so the ENTIRE M4-M6 pipeline
(states -> policies -> oracle -> metrics -> figures) runs and is tested
before paid historical data exists. Scenario definitions mirror the source
paper's Monte Carlo study (baseline / high vol / low vol / bull / bear) so
M6 can reproduce its comparisons with our validated accounting.

Real data drops in by replacing this source with CachedChainSource; nothing
downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from gamma_exit.conventions import TRADING_DAYS_PER_YEAR
from gamma_exit.pnl.engine import simulate_gbm_paths
from gamma_exit.pnl.replay import QUOTE_COLUMNS, PositionSpec
from gamma_exit.pricing.black_scholes import bs_price

PRE_HISTORY_DAYS = 252  # one year of bars for the entry-time forecast


@dataclass(frozen=True)
class Scenario:
    """One synthetic market regime (names follow the paper's Table 4)."""

    name: str
    mu: float  # real-world drift
    sigma_real: float  # true (realized) vol
    sigma_iv: float  # constant marking/implied vol
    moneyness: float = 1.0  # S0 / K at entry
    n_days: int = 126  # sessions entry -> expiry


# Paper-style scenario set: vol gap and drift are the two axes that decide
# whether gamma scalping pays; spreads chosen to bracket the paper's ranges.
PAPER_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("baseline", mu=0.06, sigma_real=0.22, sigma_iv=0.20),
    Scenario("high_vol", mu=0.06, sigma_real=0.38, sigma_iv=0.35, moneyness=0.95),
    Scenario("low_vol", mu=0.04, sigma_real=0.11, sigma_iv=0.12),
    Scenario("bull", mu=0.25, sigma_real=0.18, sigma_iv=0.18, moneyness=0.95),
    Scenario("bear", mu=-0.20, sigma_real=0.30, sigma_iv=0.26, moneyness=1.05),
)


def synthetic_quotes(
    n_days: int = 126,
    s0: float = 100.0,
    k: float = 100.0,
    kind: str = "call",
    sigma_iv=0.20,  # scalar, or array of length n_days+1 (per-day marking vol)
    sigma_real: float = 0.30,
    mu: float = 0.06,
    r: float = 0.02,
    q: float = 0.0,
    spread: float = 0.02,
    seed: int = 314,
    start: date = date(2026, 1, 5),  # a Monday
):
    """(quotes frame, PositionSpec entry->expiry, path row vector).

    Consecutive business days; mid = BS(sigma_iv) mark on an exact GBM path
    with true vol sigma_real. The M3 equivalence gate rests on this frame
    reproducing the synthetic engine exactly.
    """
    dates = pd.bdate_range(start, periods=n_days + 1).date
    t_years = n_days / TRADING_DAYS_PER_YEAR
    path = simulate_gbm_paths(s0, mu, sigma_real, t_years, n_days, 1, seed)[0]

    tte = (n_days - np.arange(n_days + 1)) / TRADING_DAYS_PER_YEAR
    iv = np.broadcast_to(np.asarray(sigma_iv, dtype=float), (n_days + 1,))
    mid = np.array(
        [bs_price(path[j], k, tte[j], r, iv[j], q, kind) for j in range(n_days + 1)]
    )

    quotes = pd.DataFrame(
        {
            "date": dates,
            "spot": path,
            "bid": mid - 0.5 * spread,
            "ask": mid + 0.5 * spread,
            "mid": mid,
            "volume": 1000,
            "open_interest": 5000,
        }
    )[QUOTE_COLUMNS]
    spec = PositionSpec(
        underlying="SYN",
        expiry=dates[-1],
        strike=k,
        kind=kind,  # type: ignore[arg-type]
        entry_date=dates[0],
    )
    return quotes, spec, path[None, :]


@dataclass(frozen=True)
class Candidate:
    """One position ready for the runner: quotes + spec + pre-entry bars."""

    quotes: pd.DataFrame
    spec: PositionSpec
    pre_history: pd.Series  # closes indexed by date, strictly before entry
    scenario: str
    seed: int


class SyntheticSource:
    """Yields `n_per_scenario` independent positions per scenario.

    Pre-entry history is simulated with the SAME true vol as the position
    window, so the causal EWMA forecast is an honest (noisy) estimate of
    sigma_real -- exactly the information structure a real trader faces
    when realized vol persists across the entry date.
    """

    def __init__(
        self,
        scenarios: tuple[Scenario, ...] = PAPER_SCENARIOS,
        n_per_scenario: int = 40,
        r: float = 0.02,
        q: float = 0.0,
        s0: float = 100.0,
        base_seed: int = 20260702,
    ) -> None:
        self.scenarios = scenarios
        self.n_per_scenario = n_per_scenario
        self.r, self.q, self.s0 = r, q, s0
        self.base_seed = base_seed

    def positions(self) -> list[Candidate]:
        out: list[Candidate] = []
        seed = self.base_seed
        for sc in self.scenarios:
            for i in range(self.n_per_scenario):
                seed += 1
                # entry dates staggered weekly: overlapping windows share a
                # calendar, mimicking the real panel's cross-correlation
                start = date(2026, 1, 5) + timedelta(weeks=i % 26)
                start = np.busday_offset(start, 0, roll="forward").item()
                quotes, spec, _ = synthetic_quotes(
                    n_days=sc.n_days,
                    s0=self.s0,
                    k=self.s0 / sc.moneyness,
                    sigma_iv=sc.sigma_iv,
                    sigma_real=sc.sigma_real,
                    mu=sc.mu,
                    r=self.r,
                    q=self.q,
                    seed=seed,
                    start=start,
                )
                pre = self._pre_history(sc, seed)
                pre_dates = pd.bdate_range(
                    end=start - timedelta(days=1), periods=PRE_HISTORY_DAYS
                ).date
                out.append(
                    Candidate(
                        quotes=quotes,
                        spec=spec,
                        pre_history=pd.Series(pre, index=pre_dates),
                        scenario=sc.name,
                        seed=seed,
                    )
                )
        return out

    def _pre_history(self, sc: Scenario, seed: int) -> np.ndarray:
        t_years = PRE_HISTORY_DAYS / TRADING_DAYS_PER_YEAR
        path = simulate_gbm_paths(
            self.s0, sc.mu, sc.sigma_real, t_years, PRE_HISTORY_DAYS - 1, 1, seed + 900_000
        )[0]
        # scale so history ENDS at s0 (the entry spot), not starts there
        return path * (self.s0 / path[-1])
