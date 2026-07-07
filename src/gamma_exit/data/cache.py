"""Write-once Parquet cache with DuckDB SQL over it.

HARD RULE: raw provider pulls are immutable. `write` refuses to overwrite an
existing key -- re-running a pull can never silently rewrite history. If a
pull was genuinely bad, `quarantine` it (moves it aside with a reason file)
and re-pull; nothing here ever deletes data.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
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
                "(use .quarantine() if the pull was bad)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)  # atomic: readers never see a partial file
        return path

    def read(self, dataset: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path_for(dataset, key))

    def quarantine(self, dataset: str, key: str, reason: str) -> Path:
        """Move a bad pull aside (never delete) so the key can be re-pulled.

        Write-once stays intact: the original bytes survive under
        _quarantine/ with a sidecar .reason.txt recording why and when.
        """
        src = self.path_for(dataset, key)
        if not src.exists():
            raise FileNotFoundError(f"nothing to quarantine at {src}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst = self.root / "_quarantine" / dataset / f"{key}.{stamp}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        dst.with_suffix(".reason.txt").write_text(
            f"quarantined {stamp}\ndataset={dataset} key={key}\nreason: {reason}\n"
        )
        return dst

    def query(self, sql: str) -> pd.DataFrame:
        """DuckDB SQL over the cache. Reference files as
        read_parquet('{root}/<dataset>/*.parquet') via the literal {root} token.

        Literal str.replace, NOT str.format: DuckDB SQL legitimately contains
        braces (struct literals, LIKE patterns) that format() would eat.
        """
        return duckdb.sql(sql.replace("{root}", str(self.root))).df()
