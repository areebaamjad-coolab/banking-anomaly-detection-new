"""Polars-based CSV loader with schema validation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.settings import settings


REQUIRED_RAW_COLUMNS = [
    "transaction_id",
    "account_id",
    "txn_timestamp",
    "amount",
    "debit_credit",
]

REQUIRED_ENGINEERED_COLUMNS = [
    "txn_amount_30d",
    "amount_zscore_self_30d",
    "device_change_count",
    "night_txn_ratio_30d",
    "avg_balance_before_7d",
]


def load_csv(path: Path | str, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if lazy:
        return pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10_000)
    return pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000)


def validate_schema(df: pl.DataFrame) -> list[str]:
    """Return list of missing required columns."""
    return [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]


def validate_feature_columns(df: pl.DataFrame) -> list[str]:
    """Return the engineered feature columns that are still missing from the input data."""
    return [c for c in REQUIRED_ENGINEERED_COLUMNS if c not in df.columns]


def ensure_datetime(df: pl.DataFrame) -> pl.DataFrame:
    if "txn_timestamp" not in df.columns:
        return df
    dtype = df.schema.get("txn_timestamp")
    if dtype == pl.Datetime or str(dtype).startswith("Datetime"):
        return df
    return df.with_columns(
        pl.col("txn_timestamp").str.to_datetime(strict=False).alias("txn_timestamp")
    )
