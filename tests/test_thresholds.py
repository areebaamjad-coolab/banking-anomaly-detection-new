"""Tests for threshold and prediction logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring.thresholds import compute_thresholds_ae, compute_thresholds_if_lof, predict


def test_if_lof_threshold_ordering() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    th, th2 = compute_thresholds_if_lof(scores, contamination=0.1)
    assert th <= th2 or th2 >= np.mean(scores)  # tier2 is z-based


def test_ae_thresholds_positive() -> None:
    errors = np.random.default_rng(42).normal(0.5, 0.1, 100)
    th, th2 = compute_thresholds_ae(errors)
    assert th > 0
    assert th2 >= th


def test_predictions_labels() -> None:
    scores = np.array([1.0, 2.0, 10.0, 3.0])
    result = predict(scores, threshold=5.0, threshold_2=8.0)
    assert result.prediction[2] == "Anomalous"
    assert result.prediction[0] == "Normal"
    assert result.prediction_2[1] == "Normal"
    assert result.prediction_2[2] == "Anomalous"
