"""Typed, validated experiment config (eng-review decision 2A).

The YAML in configs/ is the single source of truth for run parameters; code
never carries its own copy of a number that lives here. `extra="forbid"`
makes typos in the YAML fail at load time instead of silently doing nothing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExperimentConfig(_Strict):
    name: str
    seed: int


class DataConfig(_Strict):
    provider: Literal["yfinance", "thetadata", "optionmetrics"]
    cache_root: str
    universe: list[str] = Field(min_length=1)
    start: date
    end: date


class RatesConfig(_Strict):
    mode: Literal["constant"]
    risk_free: float


class QuotesConfig(_Strict):
    price: Literal["mid"]  # mid-or-drop is a hard rule; no other value allowed
    min_volume: int
    min_open_interest: int
    moneyness_band: tuple[float, float]
    dte_buckets: list[tuple[int, int]]


class CostLevel(_Strict):
    name: str
    option_spread_frac: float  # fraction of observed half-spread (replay)
    share_cost_per_share: float  # dollars per share traded (both modes)


class CostsConfig(_Strict):
    levels: list[CostLevel] = Field(min_length=1)


class ForecastVolConfig(_Strict):
    model: Literal["ewma"]
    decay: float = Field(alias="lambda", gt=0.0, lt=1.0)


class RealizedVolConfig(_Strict):
    estimator: Literal["close_to_close", "parkinson"]


class VolConfig(_Strict):
    forecast: ForecastVolConfig
    realized: RealizedVolConfig


class ValidationScenario(_Strict):
    s0: float
    strike: float
    t_years: float
    r: float
    mu: float
    sigma_real: float
    sigma_iv: float
    n_paths: int


class Config(_Strict):
    experiment: ExperimentConfig
    data: DataConfig
    rates: RatesConfig
    quotes: QuotesConfig
    costs: CostsConfig
    vol: VolConfig
    entries: list[str]
    policies: list[str]
    validation: ValidationScenario


DEFAULT_CONFIG = Path("configs/baseline.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
