"""Tests for the single time basis and the config loader."""

from datetime import date, datetime, timezone

import pytest

from gamma_exit.config import DEFAULT_CONFIG, load_config
from gamma_exit.conventions import (
    TRADING_DAYS_PER_YEAR,
    TRADING_DT,
    trading_days_between,
    years_to_expiry,
)

# 2026-07-06 is a Monday, 2026-07-10 a Friday, 2026-07-11 a Saturday
MON, TUE, FRI, SAT = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 10), date(2026, 7, 11)


class TestConventions:
    def test_basis_constants(self):
        assert TRADING_DAYS_PER_YEAR == 252
        assert TRADING_DT == pytest.approx(1 / 252)

    def test_expiring_tomorrow_is_one_session(self):
        assert years_to_expiry(TUE, MON) == pytest.approx(1 / 252)

    def test_weekend_gap_is_one_session(self):
        # Friday close -> Monday expiry: only Monday's session remains
        fri_before = date(2026, 7, 3)
        assert years_to_expiry(MON, fri_before) == pytest.approx(1 / 252)

    def test_full_week_is_five_sessions(self):
        assert years_to_expiry(FRI, date(2026, 7, 3)) == pytest.approx(5 / 252)

    def test_zero_dte_is_half_session_never_negative(self):
        assert years_to_expiry(MON, MON) == pytest.approx(0.5 / 252)
        assert years_to_expiry(MON, TUE) == 0.0  # already expired

    def test_saturday_expiry_counts_no_phantom_session(self):
        # Friday asof -> Saturday expiry: no trading sessions remain
        assert years_to_expiry(SAT, FRI) == 0.0

    def test_datetime_asof_accepted(self):
        asof = datetime(2026, 7, 6, 20, 0, tzinfo=timezone.utc)
        assert years_to_expiry(TUE, asof) == pytest.approx(1 / 252)

    def test_trading_days_between_never_negative(self):
        assert trading_days_between(TUE, MON) == 0
        assert trading_days_between(MON, FRI) == 4


class TestConfig:
    def test_baseline_loads_and_is_typed(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.experiment.seed == 20260702
        assert cfg.quotes.price == "mid"
        assert cfg.vol.forecast.decay == pytest.approx(0.94)  # yaml key: lambda
        assert [c.name for c in cfg.costs.levels] == ["zero", "half_spread", "full_spread"]
        assert "unconditional" in cfg.entries  # 13A control arm pre-registered
        assert "oracle" in cfg.policies

    def test_typo_keys_fail_loudly(self, tmp_path):
        import yaml

        raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
        raw["rates"]["risk_freee"] = 0.05  # typo must not be silently ignored
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(Exception, match="risk_freee"):
            load_config(bad)

    def test_cost_levels_cover_all_three_regimes(self):
        # hard rule: every backtest runs at zero / half / full spread
        cfg = load_config(DEFAULT_CONFIG)
        fracs = sorted(c.option_spread_frac for c in cfg.costs.levels)
        assert fracs == [0.0, 0.5, 1.0]
