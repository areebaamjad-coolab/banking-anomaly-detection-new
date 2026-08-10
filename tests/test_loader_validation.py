import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import validate_feature_columns


def test_validate_feature_columns_requires_engineered_columns() -> None:
    df = pl.DataFrame({
        "transaction_id": ["1"],
        "account_id": ["A1"],
        "txn_timestamp": ["2024-01-01"],
        "amount": [100.0],
        "debit_credit": ["debit"],
        "channel": ["ATM"],
    })

    missing = validate_feature_columns(df)
    assert "txn_amount_30d" in missing
    assert "amount_zscore_self_30d" in missing
    assert "device_change_count" in missing
