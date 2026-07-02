"""Offline tests for the canonical schema and the write-once cache."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from gamma_exit.data.cache import CacheKeyError, WriteOnceCache
from gamma_exit.data.schema import OPTION_COLUMNS, OptionRecord, options_to_frame

ASOF = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)


def _rec(**kw) -> OptionRecord:
    base = dict(
        asof=ASOF, provider="test", underlying="SPY", expiry=date(2026, 8, 21),
        strike=100.0, kind="call", bid=1.0, ask=1.2,
    )
    return OptionRecord(**{**base, **kw})


class TestSchema:
    def test_mid_requires_sane_two_sided_quote(self):
        assert _rec().mid == pytest.approx(1.1)
        assert _rec(bid=None).mid is None
        assert _rec(bid=0.0).mid is None  # zero bid = not a real market
        assert _rec(bid=2.0, ask=1.0).mid is None  # crossed

    def test_naive_asof_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _rec(asof=datetime(2026, 7, 1))

    def test_frame_has_canonical_columns(self):
        df = options_to_frame([_rec()])
        assert list(df.columns) == OPTION_COLUMNS


class TestWriteOnceCache:
    def test_round_trip(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        df = options_to_frame([_rec()])
        cache.write(df, "chains", "SPY/2026-07-01")
        out = cache.read("chains", "SPY/2026-07-01")
        pd.testing.assert_frame_equal(out, df)

    def test_second_write_to_same_key_raises(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        df = options_to_frame([_rec()])
        cache.write(df, "chains", "SPY/2026-07-01")
        with pytest.raises(FileExistsError, match="write-once"):
            cache.write(df, "chains", "SPY/2026-07-01")
        # and the original bytes are untouched
        pd.testing.assert_frame_equal(cache.read("chains", "SPY/2026-07-01"), df)

    def test_unsafe_keys_rejected(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        for bad in ("../escape", "a b", "x;y"):
            with pytest.raises(CacheKeyError):
                cache.path_for("chains", bad)

    def test_duckdb_query_over_cache(self, tmp_path):
        cache = WriteOnceCache(tmp_path)
        cache.write(options_to_frame([_rec(), _rec(kind="put")]), "chains", "SPY/x")
        out = cache.query(
            "select kind, count(*) n from read_parquet('{root}/chains/**/*.parquet') group by 1"
        )
        assert sorted(out["kind"]) == ["call", "put"]
