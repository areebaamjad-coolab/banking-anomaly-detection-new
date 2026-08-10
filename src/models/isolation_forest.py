"""Isolation Forest with hyperparameter grid search."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from config.settings import settings


def grid_search_if(X_train: np.ndarray) -> tuple[IsolationForest, dict]:
    best_model = None
    best_params = None
    best_score = -np.inf

    grid = list(product(
        settings.if_contamination,
        settings.if_n_estimators,
        settings.if_max_features,
    ))

    for contamination, n_estimators, max_features in grid:
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=settings.if_max_samples,
            random_state=settings.random_state,
            n_jobs=-1,
        )
        model.fit(X_train)
        scores = -model.score_samples(X_train)
        preds = model.predict(X_train)
        pseudo = (preds == -1).astype(int)

        flagged_rate = pseudo.mean()
        align = 1.0 - abs(flagged_rate - contamination) / max(contamination, 1e-9)

        try:
            sil = silhouette_score(X_train, pseudo) if len(set(pseudo)) > 1 else 0.0
        except Exception:
            sil = 0.0

        combined = align * 0.5 + (sil + 1) * 0.25 + 0.25
        if combined > best_score:
            best_score = combined
            best_model = model
            best_params = {
                "contamination": contamination,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_samples": settings.if_max_samples,
            }

    return best_model, best_params  # type: ignore


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)


def save_model(model: IsolationForest, params: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    (path / "params.json").write_text(json.dumps(params), encoding="utf-8")


def load_model(path: Path) -> tuple[IsolationForest, dict]:
    model = joblib.load(path / "model.joblib")
    params = json.loads((path / "params.json").read_text(encoding="utf-8"))
    return model, params
