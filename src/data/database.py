"""
Local CSV database — single master file combining:
  raw transactions + engineered features + customer profile + demographics

All reads/writes use Polars.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

from config.settings import Settings, settings


class LocalDatabase:
    """File-based local database backed by a single CSV (Polars)."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self.path = self.cfg.database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
        if not self.exists():
            raise FileNotFoundError(
                f"Database not found at {self.path}. "
                "Run generate-synthetic or import-csv first."
            )
        if lazy:
            return pl.scan_csv(self.path, try_parse_dates=True, infer_schema_length=10_000)
        return pl.read_csv(self.path, try_parse_dates=True, infer_schema_length=10_000)

    def write(self, df: pl.DataFrame, backup: bool = True) -> Path:
        if backup and self.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.path.with_suffix(f".backup_{ts}.csv")
            shutil.copy2(self.path, backup_path)
        df.write_csv(self.path)
        return self.path

    def import_csv(self, source: Path, overwrite: bool = False) -> pl.DataFrame:
        """Import external CSV into the local database."""
        if self.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists. Pass overwrite=True to replace.")
        df = pl.read_csv(source, try_parse_dates=True, infer_schema_length=10_000)
        self.write(df, backup=False)
        return df

    def append_csv(self, source: Path) -> pl.DataFrame:
        """Append rows from another CSV (same schema)."""
        new_df = pl.read_csv(source, try_parse_dates=True, infer_schema_length=10_000)
        if self.exists():
            existing = self.read()
            combined = pl.concat([existing, new_df], how="diagonal_relaxed")
            self.write(combined)
            return combined
        self.write(new_df, backup=False)
        return new_df

    def schema(self) -> dict[str, pl.DataType]:
        if not self.exists():
            return {}
        return dict(self.read().schema)

    def row_count(self) -> int:
        if not self.exists():
            return 0
        return self.read().height

    def account_count(self) -> int:
        if not self.exists():
            return 0
        df = self.read()
        if "account_id" not in df.columns:
            return 0
        return df.select("account_id").n_unique()

    def summary(self) -> dict:
        if not self.exists():
            return {"exists": False, "path": str(self.path)}
        df = self.read()
        return {
            "exists": True,
            "path": str(self.path),
            "rows": df.height,
            "columns": len(df.columns),
            "accounts": df.select("account_id").n_unique() if "account_id" in df.columns else None,
            "date_range": self._date_range(df),
        }

    @staticmethod
    def _date_range(df: pl.DataFrame) -> dict | None:
        if "txn_timestamp" not in df.columns:
            return None
        ts = df.select(
            pl.col("txn_timestamp").str.to_datetime(strict=False).alias("ts")
        ).drop_nulls()
        if ts.is_empty():
            return None
        return {
            "min": str(ts["ts"].min()),
            "max": str(ts["ts"].max()),
        }

    def query_accounts(self, account_ids: list[str]) -> pl.DataFrame:
        df = self.read()
        return df.filter(pl.col("account_id").is_in(account_ids))

    def query_train_test(self, split: str) -> pl.DataFrame:
        df = self.read()
        if "Train/Test" not in df.columns:
            raise ValueError("Train/Test column missing. Run train pipeline first.")
        return df.filter(pl.col("Train/Test") == split)
