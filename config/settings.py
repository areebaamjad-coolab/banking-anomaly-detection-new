"""Application settings loaded from .env and feature registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@dataclass
class Settings:
    root: Path = ROOT
    database_dir: Path = field(default_factory=lambda: ROOT / os.getenv("DATABASE_DIR", "data/database"))
    database_csv: str = os.getenv("DATABASE_CSV", "transactions_master.csv")
    output_dir: Path = field(default_factory=lambda: ROOT / os.getenv("OUTPUT_DIR", "data/outputs"))
    model_dir: Path = field(default_factory=lambda: ROOT / os.getenv("MODEL_DIR", "data/processed/models"))

    train_ratio: float = float(os.getenv("TRAIN_RATIO", "0.8"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

    z_threshold: float = float(os.getenv("Z_THRESHOLD", "±3.5"))
    z_threshold_tiers: list[float] = field(
        default_factory=lambda: _parse_float_list(os.getenv("Z_THRESHOLD_TIERS", "±3.0,±4.0,±5.0"))
    )

    if_contamination: list[float] = field(
        default_factory=lambda: _parse_float_list(os.getenv("IF_CONTAMINATION", "0.001,0.002,0.003,0.005,0.010"))
    )
    if_n_estimators: list[int] = field(
        default_factory=lambda: _parse_int_list(os.getenv("IF_N_ESTIMATORS", "100,300,500"))
    )
    if_max_features: list[float] = field(
        default_factory=lambda: _parse_float_list(os.getenv("IF_MAX_FEATURES", "0.6,0.8,1.0"))
    )
    if_max_samples: str = os.getenv("IF_MAX_SAMPLES", "auto")

    lof_n_neighbors: int = int(os.getenv("LOF_N_NEIGHBORS", "20"))
    lof_contamination: float = float(os.getenv("LOF_CONTAMINATION", "0.005"))

    ocsvm_nu: float = float(os.getenv("OCSVM_NU", "0.01"))
    ocsvm_gamma: str = os.getenv("OCSVM_GAMMA", "scale")

    ae_epochs: int = int(os.getenv("AE_EPOCHS", "50"))
    ae_batch_size: int = int(os.getenv("AE_BATCH_SIZE", "256"))
    ae_encoding_dim: int = int(os.getenv("AE_ENCODING_DIM", "32"))
    ae_learning_rate: float = float(os.getenv("AE_LEARNING_RATE", "0.001"))

    feature_registry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        registry_path = self.root / "config" / "feature_registry.yaml"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                self.feature_registry = yaml.safe_load(f) or {}

    @property
    def database_path(self) -> Path:
        return self.database_dir / self.database_csv

    @property
    def excluded_from_model(self) -> set[str]:
        return set(self.feature_registry.get("excluded_from_model", []))

    @property
    def ready_numeric_features(self) -> list[str]:
        return list(self.feature_registry.get("ready_numeric_features", []))

    @property
    def transaction_categoricals(self) -> list[str]:
        return list(self.feature_registry.get("transaction_categoricals", []))

    @property
    def demographic_categorical(self) -> list[str]:
        return list(self.feature_registry.get("demographic_categorical", []))

    @property
    def feature_categories(self) -> dict[str, dict[str, list[str]]]:
        categories = self.feature_registry.get("feature_categories", {})
        return {
            name: {
                key: list(values)
                for key, values in details.items()
            }
            for name, details in categories.items()
        }


settings = Settings()
