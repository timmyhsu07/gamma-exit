"""Write-once Parquet cache with DuckDB SQL over it.

HARD RULE (PROJECT_BRIEF.md section 2): raw provider pulls are immutable.
`write` refuses to overwrite an existing key -- re-running a pull can never
silently rewrite history. If a pull was genuinely bad, delete the file by
hand and re-pull; that action should be loud and deliberate.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd


class CacheKeyError(ValueError):
    pass


class WriteOnceCache:
    def __init__(self, root: str | Path = "cache") -> None:
        self.root = Path(root)

    def path_for(self, dataset: str, key: str) -> Path:
        for part in (dataset, key):
            if not re.fullmatch(r"[A-Za-z0-9._\-/]+", part) or ".." in part:
                raise CacheKeyError(f"unsafe cache key component: {part!r}")
        return self.root / dataset / f"{key}.parquet"

    def exists(self, dataset: str, key: str) -> bool:
        return self.path_for(dataset, key).exists()

    def write(self, df: pd.DataFrame, dataset: str, key: str) -> Path:
        """Write once; raises FileExistsError if the key already exists."""
        path = self.path_for(dataset, key)
        if path.exists():
            raise FileExistsError(
                f"cache is write-once: {path} already exists "
                "(delete manually if the pull was bad)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)  # atomic: readers never see a partial file
        return path

    def read(self, dataset: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path_for(dataset, key))

    def query(self, sql: str) -> pd.DataFrame:
        """DuckDB SQL over the cache. Reference files as
        read_parquet('<root>/<dataset>/*.parquet') via the {root} placeholder."""
        return duckdb.sql(sql.format(root=str(self.root))).df()
