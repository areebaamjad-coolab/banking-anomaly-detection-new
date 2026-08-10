"""Feature column lists loaded from feature_registry.yaml."""

from __future__ import annotations

from config.settings import settings


def excluded_columns() -> set[str]:
    return settings.excluded_from_model


def ready_numeric() -> list[str]:
    return settings.ready_numeric_features


def transaction_categoricals() -> list[str]:
    return settings.transaction_categoricals


def demographic_categorical() -> list[str]:
    return settings.demographic_categorical


def pending_groups() -> list[str]:
    return list(settings.feature_registry.get("pending_feature_groups", []))


def feature_groups(df_columns: list[str]) -> dict[str, list[str]]:
    """Return present feature columns grouped by category."""
    excluded = excluded_columns()
    grouped: dict[str, list[str]] = {}

    for category_name, category_definition in settings.feature_categories.items():
        grouped[category_name] = []
        for feature_type in ("numeric", "categorical"):
            for column in category_definition.get(feature_type, []):
                if column in df_columns and column not in excluded:
                    grouped[category_name].append(column)

    grouped["numeric"] = [
        c for c in ready_numeric() if c in df_columns and c not in excluded
    ]
    grouped["transaction_categorical"] = [
        c for c in transaction_categoricals() if c in df_columns and c not in excluded
    ]
    grouped["demographic_categorical"] = [
        c for c in demographic_categorical() if c in df_columns and c not in excluded
    ]
    return grouped


def all_model_features(df_columns: list[str]) -> list[str]:
    """Return feature columns present in df, excluding IDs and pending groups."""
    groups = feature_groups(df_columns)
    return [
        feature
        for category_name in [
            "amount_and_balance",
            "device_and_channel_network_and_beneficiary",
            "velocity_and_timing",
            "demographic_and_customer_profile",
        ]
        for feature in groups.get(category_name, [])
    ]
