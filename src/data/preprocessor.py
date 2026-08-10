"""Preprocessing: cleaning, txn_timestamp handling, sorting, partitioning, derived columns."""

from __future__ import annotations

import re
from datetime import datetime

import polars as pl

from config.settings import settings


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize and filter raw transaction rows before feature engineering."""
    if df.is_empty():
        return df

    cleaned = df.clone()

    if "transaction_id" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("transaction_id").cast(pl.String).str.strip_chars().alias("transaction_id")
        )
    if "account_id" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("account_id").cast(pl.String).str.strip_chars().alias("account_id")
        )

    if "debit_credit" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("debit_credit")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .alias("debit_credit")
        )
        cleaned = cleaned.with_columns(
            pl.when(pl.col("debit_credit").is_in(["debit", "credit"]))
            .then(pl.col("debit_credit"))
            .otherwise(None)
            .alias("debit_credit")
        )

    if "amount" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("amount")
            .map_elements(_parse_amount, return_dtype=pl.Float64)
            .alias("amount")
        )

    if "txn_timestamp" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("txn_timestamp")
            .map_elements(_parse_timestamp_value, return_dtype=pl.Datetime("us"))
            .alias("txn_timestamp")
        )

    required = [c for c in ["transaction_id", "account_id", "amount", "debit_credit", "txn_timestamp"] if c in cleaned.columns]
    if required:
        cleaned = cleaned.drop_nulls(subset=required)
    return cleaned


def preprocess(df: pl.DataFrame, train_ratio: float | None = None) -> pl.DataFrame:
    train_ratio = train_ratio or settings.train_ratio
    df = clean_data(df)
    df = _parse_timestamp(df)
    df = df.sort(["account_id", "txn_timestamp", "transaction_id"])
    df = _add_time_columns(df)
    df = _add_daily_aggregates(df)
    df = _partition_by_account(df, train_ratio)
    return df


def _parse_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    if "txn_timestamp" not in df.columns:
        return df
    df = df.with_columns(
        pl.col("txn_timestamp").cast(pl.Datetime("us"), strict=False).alias("txn_timestamp")
    )
    return df.filter(pl.col("txn_timestamp").is_not_null())


def _parse_amount(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = re.sub(r"[^0-9.\-]", "", text)
        if not text or text in {"-", "."}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp_value(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _add_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col("txn_timestamp").dt.date().alias("post_date"),
        pl.col("txn_timestamp").dt.time().alias("time_of_transaction"),
        pl.col("txn_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S").alias("date_time"),
        (pl.col("txn_timestamp").dt.epoch(time_unit="s")).alias("unix_timestamp"),
        pl.col("txn_timestamp").dt.year().alias("transaction_year"),
    ]
    df = df.with_columns(exprs)
    df = df.with_columns(
        (
            pl.col("txn_timestamp").dt.epoch(time_unit="s")
            - pl.col("txn_timestamp").dt.epoch(time_unit="s").shift(1).over("account_id")
        ).alias("timestamp_interval")
    )
    return df


def _add_daily_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    if "balance_after" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("balance_after"))

    daily = (
        df.group_by(["account_id", "post_date"])
        .agg([
            pl.when(pl.col("debit_credit") == "debit")
            .then(pl.col("amount"))
            .otherwise(0.0)
            .sum()
            .alias("daily_debit_amount"),
            pl.when(pl.col("debit_credit") == "credit")
            .then(pl.col("amount"))
            .otherwise(0.0)
            .sum()
            .alias("daily_credit_amount"),
            pl.len().alias("daily_count"),
            pl.col("balance_after").last().alias("daily_balance"),
        ])
    )

    return df.join(daily, on=["account_id", "post_date"], how="left")


def _partition_by_account(df: pl.DataFrame, train_ratio: float) -> pl.DataFrame:
    acct_first = (
        df.group_by("account_id")
        .agg(pl.col("txn_timestamp").min().alias("first_ts"))
        .sort("first_ts")
    )
    n_train = max(1, int(acct_first.height * train_ratio))
    train_accounts = acct_first.head(n_train)["account_id"].to_list()

    return df.with_columns(
        pl.when(pl.col("account_id").is_in(train_accounts))
        .then(pl.lit("Train"))
        .otherwise(pl.lit("Test"))
        .alias("Train/Test")
    )


def get_train_test(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("Train/Test") == "Train")
    test = df.filter(pl.col("Train/Test") == "Test")
    return train, test
