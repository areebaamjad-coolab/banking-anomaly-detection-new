"""Local Outlier Factor model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from config.settings import settings


def train_lof(X_train: np.ndarray) -> LocalOutlierFactor:
    model = LocalOutlierFactor(
        n_neighbors=min(settings.lof_n_neighbors, len(X_train) - 1),
        contamination=settings.lof_contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)


def save_model(model: LocalOutlierFactor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    params = {"contamination": settings.lof_contamination, "n_neighbors": settings.lof_n_neighbors}
    (path / "params.json").write_text(json.dumps(params), encoding="utf-8")


def load_model(path: Path) -> LocalOutlierFactor:
    return joblib.load(path / "model.joblib")
