"""One-Class SVM anomaly detector."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import OneClassSVM

from config.settings import settings


def train_ocsvm(X_train: np.ndarray) -> OneClassSVM:
    model = OneClassSVM(
        kernel="rbf",
        nu=settings.ocsvm_nu,
        gamma=settings.ocsvm_gamma,
        cache_size=1024,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def save_model(model: OneClassSVM, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    params = {"nu": settings.ocsvm_nu, "gamma": settings.ocsvm_gamma}
    (path / "params.json").write_text(json.dumps(params), encoding="utf-8")


def load_model(path: Path) -> OneClassSVM:
    return joblib.load(path / "model.joblib")
