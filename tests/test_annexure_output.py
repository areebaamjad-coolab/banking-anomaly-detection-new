"""Tests for Annexure D column order."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

polars = pytest.importorskip("polars")

from src.output.annexure_d import ANNEXURE_COLUMNS, build_annexure_d


def test_annexure_column_order() -> None:
    df = polars.DataFrame({
        "account_id": ["A1"],
        "branch_code": ["B1"],
        "account_open_date": ["2020-01-01"],
        "post_date": [polars.date(2024, 1, 1)],
        "time_of_transaction": ["10:00:00"],
        "amount": [100.0],
        "debit_credit": ["debit"],
        "daily_balance": [900.0],
        "transaction_id": ["T1"],
        "transaction_type_desc": ["TRANSFER"],
        "channel": ["ATM"],
        "debit_acct_no": ["A1"],
        "credit_acct_no": ["X1"],
        "transaction_type": ["AC"],
        "transaction_year": [2024],
        "Train/Test": ["Train"],
        "date_time": ["2024-01-01 10:00:00"],
        "unix_timestamp": [1704110400],
        "timestamp_interval": [0.0],
        "daily_debit_amount": [100.0],
        "daily_credit_amount": [0.0],
        "daily_count": [1],
    })
    out = build_annexure_d(df, [0.5], 0.3, 0.5, ["Normal"], ["Normal"])
    assert list(out.columns) == [c for c in ANNEXURE_COLUMNS if c in out.columns]
    assert out.columns[0] == "Account #"
    assert "Prediction" in out.columns
    assert "Threshold_2" in out.columns
