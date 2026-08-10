import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.column_lists import all_model_features, feature_groups


def test_feature_groups_include_requested_categories() -> None:
    df_columns = [
        "txn_amount_30d",
        "avg_balance_before_7d",
        "channel",
        "beneficiary_account",
        "beneficiary_bank_issuer",
        "night_txn_ratio_30d",
        "amount_zscore_self_30d",
        "avg_amount_to_balance_ratio_30d",
        "device_change_count",
        "primary_device_used",
        "customer_type",
        "occupation",
        "account_age_days",
        "transaction_id",
    ]

    groups = feature_groups(df_columns)

    assert "amount_and_balance" in groups
    assert "device_and_channel_network_and_beneficiary" in groups
    assert "velocity_and_timing" in groups
    assert "demographic_and_customer_profile" in groups

    assert "txn_amount_30d" in groups["amount_and_balance"]
    assert "channel" in groups["device_and_channel_network_and_beneficiary"]
    assert "beneficiary_account" in groups["device_and_channel_network_and_beneficiary"]
    assert "device_change_count" in groups["device_and_channel_network_and_beneficiary"]
    assert "primary_device_used" in groups["device_and_channel_network_and_beneficiary"]
    assert "amount_zscore_self_30d" in groups["amount_and_balance"]
    assert "avg_amount_to_balance_ratio_30d" in groups["amount_and_balance"]
    assert "night_txn_ratio_30d" in groups["velocity_and_timing"]
    assert "customer_type" in groups["demographic_and_customer_profile"]
    assert "occupation" in groups["demographic_and_customer_profile"]

    model_features = all_model_features(df_columns)
    assert "txn_amount_30d" in model_features
    assert "channel" in model_features
    assert "customer_type" in model_features
    assert "beneficiary_account" in model_features
    assert "amount_zscore_self_30d" in model_features
    assert "device_change_count" in model_features
    assert "primary_device_used" in model_features
