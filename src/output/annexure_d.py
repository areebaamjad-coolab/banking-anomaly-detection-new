"""Annexure D output builder and exporter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

ANNEXURE_COLUMNS = [
    "Account #",
    "Branch",
    "Account Opening Date",
    "Post Date",
    "Time of Transaction",
    "Debit",
    "Credit",
    "Daily Balance",
    "Transaction Reference",
    "Txn Type Description",
    "Mode of Transaction",
    "Debit Account #",
    "Credit Account #",
    "Transaction Code",
    "Account Opening Year",
    "Total Transaction Count",
    "Transaction Year",
    "Train/Test",
    "Date Time",
    "Unix Timestamp",
    "Timestamp Interval",
    "Daily Debit Amount",
    "Daily Credit Amount",
    "Daily Count",
    "Prediction",
    "Reconstruction Error",
    "Threshold",
    "Threshold_2",
    "Prediction_2",
]


def build_annexure_d(
    df: pl.DataFrame,
    scores: list[float],
    threshold: float,
    threshold_2: float,
    predictions: list[str],
    predictions_2: list[str],
) -> pl.DataFrame:
    n = df.height
    debit = pl.when(pl.col("debit_credit") == "debit").then(pl.col("amount")).otherwise(None)
    credit = pl.when(pl.col("debit_credit") == "credit").then(pl.col("amount")).otherwise(None)

    acct_open = pl.col("account_open_date") if "account_open_date" in df.columns else pl.lit(None)
    acct_open_year = (
        pl.col("account_open_date").str.slice(0, 4).cast(pl.Int64)
        if "account_open_date" in df.columns
        else pl.lit(None)
    )
    total_txn = pl.col("transaction_id").count().over("account_id")

    debit_acct = pl.col("debit_acct_no").alias("Debit Account #") if "debit_acct_no" in df.columns else pl.lit(None).alias("Debit Account #")
    credit_acct = pl.col("credit_acct_no").alias("Credit Account #") if "credit_acct_no" in df.columns else pl.lit(None).alias("Credit Account #")

    out = df.with_columns([
        pl.col("account_id").alias("Account #"),
        pl.col("branch_code").alias("Branch"),
        acct_open.alias("Account Opening Date"),
        pl.col("post_date").alias("Post Date"),
        pl.col("time_of_transaction").cast(pl.Utf8).alias("Time of Transaction"),
        debit.alias("Debit"),
        credit.alias("Credit"),
        pl.col("daily_balance").alias("Daily Balance"),
        pl.col("transaction_id").alias("Transaction Reference"),
        pl.col("transaction_type_desc").alias("Txn Type Description"),
        pl.col("channel").alias("Mode of Transaction"),
        debit_acct,
        credit_acct,
        pl.col("transaction_type").alias("Transaction Code"),
        acct_open_year.alias("Account Opening Year"),
        total_txn.alias("Total Transaction Count"),
        pl.col("transaction_year").alias("Transaction Year"),
        pl.col("Train/Test"),
        pl.col("date_time").alias("Date Time"),
        pl.col("unix_timestamp").alias("Unix Timestamp"),
        pl.col("timestamp_interval").alias("Timestamp Interval"),
        pl.col("daily_debit_amount").alias("Daily Debit Amount"),
        pl.col("daily_credit_amount").alias("Daily Credit Amount"),
        pl.col("daily_count").alias("Daily Count"),
        pl.Series("Prediction", predictions),
        pl.Series("Reconstruction Error", scores),
        pl.lit(threshold).alias("Threshold"),
        pl.lit(threshold_2).alias("Threshold_2"),
        pl.Series("Prediction_2", predictions_2),
    ])

    present = [c for c in ANNEXURE_COLUMNS if c in out.columns]
    return out.select(present)


def export_annexure(df: pl.DataFrame, model_name: str, output_dir: Path) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"annexure_d_{model_name}_{ts}.csv"
    xlsx_path = output_dir / f"annexure_d_{model_name}_{ts}.xlsx"
    df.write_csv(csv_path)
    try:
        df.write_excel(xlsx_path)
        return csv_path, xlsx_path
    except Exception:
        try:
            import pandas as pd
            pd.DataFrame(df.to_dicts()).to_excel(xlsx_path, index=False, engine="openpyxl")
            return csv_path, xlsx_path
        except Exception:
            return csv_path, None
