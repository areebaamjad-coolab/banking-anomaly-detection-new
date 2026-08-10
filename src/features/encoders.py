"""Build model feature matrix from Polars DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.settings import settings
from src.features.column_lists import all_model_features, feature_groups


class FeatureMatrixBuilder:
    def __init__(self, selected_groups: list[str] | None = None) -> None:
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []
        self.feature_groups: dict[str, list[str]] = {}
        self.selected_groups = selected_groups or [
            "amount_and_balance",
            "device_and_channel_network_and_beneficiary",
            "velocity_and_timing",
            "demographic_and_customer_profile",
        ]
        self._fitted = False

    def _available_features(self, df: pl.DataFrame) -> list[str]:
        groups = feature_groups(df.columns)
        self.feature_groups = {
            name: groups.get(name, [])
            for name in self.selected_groups
            if name in groups
        }
        return [
            column
            for group in self.selected_groups
            for column in self.feature_groups.get(group, [])
        ]

    def fit_transform(self, train_df: pl.DataFrame) -> np.ndarray:
        self.feature_columns = self._available_features(train_df)
        if not self.feature_columns:
            raise ValueError(
                "No usable model features were found in the input data. Ensure the SQL-engineered feature columns from the feature registry are present."
            )
        X = self._encode(train_df, fit=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureMatrixBuilder not fitted.")
        X = self._encode(df, fit=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def _encode(self, df: pl.DataFrame, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []
        numeric_cols = [c for c in self.feature_columns if c in settings.ready_numeric_features]
        if numeric_cols:
            num = df.select(numeric_cols).cast(pl.Float64).to_numpy()
            parts.append(num)

        cat_cols = [c for c in self.feature_columns if c not in settings.ready_numeric_features]
        for col in cat_cols:
            series = df[col].fill_null("UNKNOWN").cast(pl.Utf8).to_list()
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(series)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                safe = [s if s in known else "UNKNOWN" for s in series]
                if "UNKNOWN" not in known:
                    safe = [s if s in known else le.classes_[0] for s in series]
                encoded = le.transform(safe)
            parts.append(encoded.reshape(-1, 1).astype(float))

        return np.hstack(parts) if parts else np.empty((df.height, 0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_columns": self.feature_columns,
            "feature_groups": self.feature_groups,
            "selected_groups": self.selected_groups,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "label_encoders": {
                k: {"classes": v.classes_.tolist()} for k, v in self.label_encoders.items()
            },
        }
        path.write_text(json.dumps(state), encoding="utf-8")

    def load(self, path: Path) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))
        self.feature_columns = state["feature_columns"]
        self.feature_groups = state.get("feature_groups", {})
        self.selected_groups = state.get("selected_groups", self.selected_groups)
        self.scaler.mean_ = np.array(state["scaler_mean"])
        self.scaler.scale_ = np.array(state["scaler_scale"])
        self.scaler.n_features_in_ = len(state["scaler_mean"])
        for col, enc in state["label_encoders"].items():
            le = LabelEncoder()
            le.classes_ = np.array(enc["classes"])
            self.label_encoders[col] = le
        self._fitted = True
