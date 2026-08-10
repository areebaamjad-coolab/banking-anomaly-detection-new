"""Tests for banking anomaly detection pipeline."""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.data.preprocessor import clean_data, get_train_test, preprocess
from src.features.encoders import FeatureMatrixBuilder


@pytest.fixture
def sample_df() -> pl.DataFrame:
    from src.data.synthetic import generate_master_csv
    return generate_master_csv(n_accounts=10, txns_per_account=20, save=False)


def test_account_partition_no_overlap(sample_df: pl.DataFrame) -> None:
    train, test = get_train_test(sample_df)
    train_accts = set(train["account_id"].to_list())
    test_accts = set(test["account_id"].to_list())
    assert train_accts.isdisjoint(test_accts)


def test_txn_timestamp_not_in_features(sample_df: pl.DataFrame) -> None:
    train, _ = get_train_test(sample_df)
    builder = FeatureMatrixBuilder()
    builder.fit_transform(train)
    assert "txn_timestamp" not in builder.feature_columns
    assert "account_id" not in builder.feature_columns


def test_feature_matrix_builder_requires_sql_engineered_columns() -> None:
    df = pl.DataFrame({
        "foo": [1],
        "bar": [2],
        "baz": [3],
    })

    builder = FeatureMatrixBuilder()
    with pytest.raises(ValueError, match="SQL-engineered feature columns"):
        builder.fit_transform(df)


def test_excluded_columns_not_in_registry_features() -> None:
    excluded = settings.excluded_from_model
    for col in excluded:
        assert col not in settings.ready_numeric_features


def test_feature_matrix_builder_groups_features(sample_df: pl.DataFrame) -> None:
    train, _ = get_train_test(sample_df)
    builder = FeatureMatrixBuilder(selected_groups=["numeric", "transaction_categorical"])
    X = builder.fit_transform(train)

    assert builder.feature_groups["numeric"]
    assert builder.feature_groups["transaction_categorical"]
    assert "demographic_categorical" not in builder.feature_groups
    assert X.shape[1] == len(builder.feature_columns)


def test_clean_data_normalizes_and_drops_bad_rows() -> None:
    df = pl.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "account_id": ["A1", "A2", "A3"],
        "txn_timestamp": ["2024-01-01 10:00:00", "bad-date", "2024-01-03 10:00:00"],
        "amount": ["100.50", "$200", "50.00"],
        "debit_credit": [" Debit ", "CREDIT", "credit"],
    })

    cleaned = clean_data(df)

    assert cleaned.height == 2
    assert cleaned["debit_credit"].to_list() == ["debit", "credit"]
    assert cleaned["amount"].to_list() == [100.5, 50.0]
    assert cleaned["txn_timestamp"].dtype == pl.Datetime
