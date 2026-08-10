"""End-to-end training and prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from config.settings import settings
from src.data.database import LocalDatabase
from src.data.feature_engineering import engineer_features
from src.data.loader import validate_feature_columns
from src.data.preprocessor import clean_data, get_train_test, preprocess
from src.evaluation.metrics import (
    evaluate_model,
    model_centrality,
    model_centrality_hits,
    udr,
    write_report,
)
from src.features.encoders import FeatureMatrixBuilder
from src.models import autoencoder as ae_mod
from src.models import isolation_forest as if_mod
from src.models import lof as lof_mod
from src.models import one_class_svm as ocsvm_mod
from src.output.annexure_d import build_annexure_d, export_annexure
from src.scoring.thresholds import compute_thresholds_ae, compute_thresholds_if_lof, predict


def load_and_prepare() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    db = LocalDatabase()
    df = db.read()
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    df = clean_data(df)
    missing_features = validate_feature_columns(df)
    if missing_features:
        raise ValueError(
            "Missing engineered feature columns from input data: " + ", ".join(missing_features)
        )
    df = engineer_features(df)
    if "Train/Test" not in df.columns:
        df = preprocess(df)
    train_df, test_df = get_train_test(df)
    return df, train_df, test_df


def _ordered_full_df(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Concatenate train then test — matches score array ordering."""
    return pl.concat([train_df, test_df], how="vertical_relaxed")


def train_all() -> dict:
    df, train_df, test_df = load_and_prepare()
    LocalDatabase().write(df)

    builder = FeatureMatrixBuilder()
    X_train = builder.fit_transform(train_df)
    X_test = builder.transform(test_df)
    builder.save(settings.model_dir / "feature_builder.json")

    full_df = _ordered_full_df(train_df, test_df)
    results: dict = {}

    if_model, if_params = if_mod.grid_search_if(X_train)
    if_mod.save_model(if_model, if_params, settings.model_dir / "if")
    results["if"] = _score_model(
        full_df, train_df, test_df, X_train, X_test,
        "isolation_forest",
        if_mod.anomaly_scores(if_model, X_train),
        if_mod.anomaly_scores(if_model, X_test),
        if_params["contamination"],
    )

    lof_model = lof_mod.train_lof(X_train)
    lof_mod.save_model(lof_model, settings.model_dir / "lof")
    results["lof"] = _score_model(
        full_df, train_df, test_df, X_train, X_test,
        "lof",
        lof_mod.anomaly_scores(lof_model, X_train),
        lof_mod.anomaly_scores(lof_model, X_test),
        settings.lof_contamination,
    )

    ocsvm_model = ocsvm_mod.train_ocsvm(X_train)
    ocsvm_mod.save_model(ocsvm_model, settings.model_dir / "ocsvm")
    results["ocsvm"] = _score_model(
        full_df, train_df, test_df, X_train, X_test,
        "ocsvm",
        ocsvm_mod.anomaly_scores(ocsvm_model, X_train),
        ocsvm_mod.anomaly_scores(ocsvm_model, X_test),
        settings.ocsvm_nu,
    )

    ae_model, losses = ae_mod.train_autoencoder(X_train)
    ae_mod.save_model(ae_model, losses, settings.model_dir / "ae")
    results["ae"] = _score_model_ae(
        full_df, train_df, test_df, X_train, X_test,
        ae_mod.reconstruction_errors(ae_model, X_train),
        ae_mod.reconstruction_errors(ae_model, X_test),
    )

    results = _append_consensus_metrics(results)
    write_report(list(results.values()), settings.output_dir)
    return results


def _append_consensus_metrics(results: dict) -> dict:
    score_map: dict[str, np.ndarray] = {}
    for name, payload in results.items():
        if "score_train" in payload:
            score_map[name] = np.array(payload["score_train"])
    if not score_map:
        return results
    contamination = settings.lof_contamination
    for name, payload in results.items():
        payload["udr"] = udr(score_map, contamination)
        payload["model_centrality"] = model_centrality(score_map, contamination)
        payload["model_centrality_hits"] = model_centrality_hits(score_map, contamination)
    return results


def _score_model(
    full_df: pl.DataFrame,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    name: str,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    contamination: float,
) -> dict:
    th, th2 = compute_thresholds_if_lof(train_scores, contamination)
    tr = predict(train_scores, th, th2)
    te = predict(test_scores, th, th2)

    full_scores = np.concatenate([train_scores, test_scores])
    full_preds = list(tr.prediction) + list(te.prediction)
    full_preds_2 = list(tr.prediction_2) + list(te.prediction_2)

    annexure = build_annexure_d(full_df, full_scores.tolist(), th, th2, full_preds, full_preds_2)
    export_annexure(annexure, name, settings.output_dir)

    pseudo = (train_scores >= th).astype(int)
    metrics = evaluate_model(
        name, train_scores, test_scores,
        list(tr.prediction), list(te.prediction),
        X_train, pseudo, contamination,
    )
    metrics["score_train"] = train_scores.tolist()
    metrics["score_test"] = test_scores.tolist()
    return metrics


def _score_model_ae(
    full_df: pl.DataFrame,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_err: np.ndarray,
    test_err: np.ndarray,
) -> dict:
    th, th2 = compute_thresholds_ae(train_err)
    tr = predict(train_err, th, th2)
    te = predict(test_err, th, th2)

    full_scores = np.concatenate([train_err, test_err])
    full_preds = list(tr.prediction) + list(te.prediction)
    full_preds_2 = list(tr.prediction_2) + list(te.prediction_2)

    annexure = build_annexure_d(full_df, full_scores.tolist(), th, th2, full_preds, full_preds_2)
    export_annexure(annexure, "autoencoder", settings.output_dir)

    pseudo = (train_err >= th).astype(int)
    metrics = evaluate_model(
        "autoencoder", train_err, test_err,
        list(tr.prediction), list(te.prediction),
        X_train, pseudo, settings.lof_contamination,
    )
    metrics["score_train"] = train_err.tolist()
    metrics["score_test"] = test_err.tolist()
    return metrics


def evaluate_all() -> Path:
    _, train_df, test_df = load_and_prepare()
    full_df = _ordered_full_df(train_df, test_df)

    builder = FeatureMatrixBuilder()
    builder.load(settings.model_dir / "feature_builder.json")
    X_train = builder.transform(train_df)
    X_test = builder.transform(test_df)

    results = []

    if_model, if_params = if_mod.load_model(settings.model_dir / "if")
    results.append(_score_model(
        full_df, train_df, test_df, X_train, X_test,
        "isolation_forest",
        if_mod.anomaly_scores(if_model, X_train),
        if_mod.anomaly_scores(if_model, X_test),
        if_params["contamination"],
    ))

    lof_model = lof_mod.load_model(settings.model_dir / "lof")
    results.append(_score_model(
        full_df, train_df, test_df, X_train, X_test,
        "lof",
        lof_mod.anomaly_scores(lof_model, X_train),
        lof_mod.anomaly_scores(lof_model, X_test),
        settings.lof_contamination,
    ))

    ocsvm_model = ocsvm_mod.load_model(settings.model_dir / "ocsvm")
    results.append(_score_model(
        full_df, train_df, test_df, X_train, X_test,
        "ocsvm",
        ocsvm_mod.anomaly_scores(ocsvm_model, X_train),
        ocsvm_mod.anomaly_scores(ocsvm_model, X_test),
        settings.ocsvm_nu,
    ))

    ae_model = ae_mod.load_model(settings.model_dir / "ae")
    results.append(_score_model_ae(
        full_df, train_df, test_df, X_train, X_test,
        ae_mod.reconstruction_errors(ae_model, X_train),
        ae_mod.reconstruction_errors(ae_model, X_test),
    ))

    return write_report(results, settings.output_dir)
