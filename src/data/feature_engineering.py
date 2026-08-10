"""Feature engineering per Feature Store Audit formulas (Polars)."""

from __future__ import annotations

import polars as pl

from config.settings import settings

WINDOWS = [7, 30, 60, 90]


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Return the input data unchanged and rely on the SQL-engineered columns from the sheet."""
    return df.sort(["account_id", "txn_timestamp"]) if "txn_timestamp" in df.columns else df
