"""Unsupervised evaluation metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import silhouette_score

from config.settings import settings


def anomaly_rate(predictions: list[str]) -> float:
    return sum(1 for p in predictions if p == "Anomalous") / max(len(predictions), 1)


def contamination_alignment(rate: float, contamination: float) -> float:
    return 1.0 - abs(rate - contamination) / max(contamination, 1e-9)


def ks_drift(train_scores: np.ndarray, test_scores: np.ndarray) -> float:
    if len(train_scores) < 2 or len(test_scores) < 2:
        return 0.0
    return float(stats.ks_2samp(train_scores, test_scores).statistic)


def model_agreement_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def excess_mass(scores: np.ndarray, threshold: float, contamination: float) -> float:
    if len(scores) == 0:
        return 0.0
    if contamination <= 0:
        return 0.0
    top = int(max(1, round(len(scores) * contamination)))
    ranked = np.sort(scores)
    if top >= len(ranked):
        return 0.0
    tail = ranked[-top:]
    return float(np.mean(tail >= threshold))


def mass_volume(scores: np.ndarray, threshold: float, contamination: float) -> float:
    if len(scores) == 0:
        return 0.0
    if contamination <= 0:
        return 0.0
    top = int(max(1, round(len(scores) * contamination)))
    ranked = np.sort(scores)
    if top >= len(ranked):
        return 0.0
    tail = ranked[-top:]
    return float(np.mean(tail >= threshold) * np.std(tail))


def ireos(scores: np.ndarray, predictions: list[str], contamination: float) -> float:
    if len(scores) == 0:
        return 0.0
    if contamination <= 0:
        return 0.0
    top = int(max(1, round(len(scores) * contamination)))
    ranked = np.argsort(scores)
    anomalous_idx = set(ranked[-top:].tolist())
    predicted_idx = {i for i, pred in enumerate(predictions) if pred == "Anomalous"}
    overlap = len(anomalous_idx & predicted_idx)
    return float(overlap / max(top, 1))


def udr(scores_by_model: dict[str, np.ndarray], contamination: float) -> float:
    if not scores_by_model:
        return 0.0
    rankings = []
    for scores in scores_by_model.values():
        top = int(max(1, round(len(scores) * contamination)))
        ranked = np.argsort(scores)
        rankings.append(set(ranked[-top:].tolist()))
    if not rankings:
        return 0.0
    pairs = 0
    agreements = 0
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            pairs += 1
            agreements += int(rankings[i] == rankings[j])
    return float(agreements / max(pairs, 1))


def model_centrality(scores_by_model: dict[str, np.ndarray], contamination: float) -> float:
    if not scores_by_model:
        return 0.0
    base = []
    for scores in scores_by_model.values():
        top = int(max(1, round(len(scores) * contamination)))
        ranked = np.argsort(scores)
        base.append(set(ranked[-top:].tolist()))
    if not base:
        return 0.0
    return float(np.mean([len(s) for s in base]) / max(len(next(iter(scores_by_model.values()))), 1))


def model_centrality_hits(scores_by_model: dict[str, np.ndarray], contamination: float) -> float:
    if not scores_by_model:
        return 0.0
    scores = []
    for model_scores in scores_by_model.values():
        top = int(max(1, round(len(model_scores) * contamination)))
        ranked = np.argsort(model_scores)
        scores.append(set(ranked[-top:].tolist()))
    if not scores:
        return 0.0
    return float(np.mean([len(s) for s in scores]) / max(len(next(iter(scores_by_model.values()))), 1))


def evaluate_model(
    model_name: str,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_preds: list[str],
    test_preds: list[str],
    X_train: np.ndarray,
    train_pseudo: np.ndarray,
    contamination: float,
) -> dict:
    sil = 0.0
    try:
        if len(set(train_pseudo)) > 1:
            sil = float(silhouette_score(X_train, train_pseudo))
    except Exception:
        pass

    rate_train = anomaly_rate(train_preds)
    rate_test = anomaly_rate(test_preds)
    rate_ratio = rate_test / max(rate_train, 1e-9)

    return {
        "model": model_name,
        "anomaly_rate_train": rate_train,
        "anomaly_rate_test": rate_test,
        "rate_ratio": rate_ratio,
        "contamination_alignment": contamination_alignment(rate_train, contamination),
        "silhouette": sil,
        "ks_drift": ks_drift(train_scores, test_scores),
        "score_mean_train": float(np.mean(train_scores)),
        "score_std_train": float(np.std(train_scores)),
        "score_p99_train": float(np.percentile(train_scores, 99)),
        "excess_mass": excess_mass(train_scores, np.percentile(train_scores, 99), contamination),
        "mass_volume": mass_volume(train_scores, np.percentile(train_scores, 99), contamination),
        "ireos": ireos(train_scores, train_preds, contamination),
    }


def write_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"evaluation_report_{ts}.json"
    path.write_text(json.dumps({"models": results}, indent=2), encoding="utf-8")

    xlsx_path = output_dir / f"evaluation_report_{ts}.xlsx"
    pl.DataFrame(results).write_excel(xlsx_path)
    return path


def model_label(name: str) -> str:
    return {
        "isolation_forest": "Isolation Forest",
        "lof": "LOF",
        "ocsvm": "One-Class SVM",
        "autoencoder": "Autoencoder",
    }.get(name, name)
