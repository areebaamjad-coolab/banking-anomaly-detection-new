"""Tests for account_id partition rules."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

polars = pytest.importorskip("polars")

from src.data.preprocessor import get_train_test, preprocess


def _make_minimal_df(n_accounts: int = 10, txns: int = 5) -> "polars.DataFrame":
    from datetime import datetime, timedelta
    rows = []
    base = datetime(2024, 1, 1)
    for a in range(n_accounts):
        for t in range(txns):
            rows.append({
                "transaction_id": f"T{a}_{t}",
                "account_id": f"ACC{a:03d}",
                "txn_timestamp": (base + timedelta(days=a, hours=t)).strftime("%Y-%m-%d %H:%M:%S"),
                "amount": 100.0 + t,
                "debit_credit": "debit",
                "branch_code": "PK0010001",
                "balance_before": 1000.0,
                "balance_after": 900.0,
                "transaction_type_desc": "TRANSFER",
                "channel": "ATM",
                "transaction_type": "AC",
            })
    df = polars.DataFrame(rows)
    return preprocess(df)


def test_no_account_overlap_between_splits() -> None:
    df = _make_minimal_df(20, 10)
    train, test = get_train_test(df)
    assert set(train["account_id"].to_list()).isdisjoint(set(test["account_id"].to_list()))


def test_train_test_column_present() -> None:
    df = _make_minimal_df(5, 3)
    assert "Train/Test" in df.columns
    assert set(df["Train/Test"].unique().to_list()) <= {"Train", "Test"}


def test_train_ratio_approximate() -> None:
    df = _make_minimal_df(100, 2)
    train, test = get_train_test(df)
    n_train_accts = train["account_id"].n_unique()
    n_test_accts = test["account_id"].n_unique()
    assert n_train_accts + n_test_accts == 100
    assert n_train_accts >= 70  # ~80%
