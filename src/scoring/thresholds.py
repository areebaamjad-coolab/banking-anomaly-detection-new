"""Threshold computation and predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import settings


@dataclass
class ThresholdResult:
    threshold: float
    threshold_2: float
    prediction: np.ndarray
    prediction_2: np.ndarray


def compute_thresholds_if_lof(
    scores: np.ndarray,
    contamination: float,
) -> tuple[float, float]:
    train_scores = scores
    threshold = float(np.percentile(train_scores, 100 * (1 - contamination)))
    mean, std = float(np.mean(train_scores)), float(np.std(train_scores))
    tier2 = settings.z_threshold_tiers[-1] if settings.z_threshold_tiers else settings.z_threshold
    threshold_2 = mean + tier2 * std
    return threshold, threshold_2


def compute_thresholds_ae(errors: np.ndarray) -> tuple[float, float]:
    mean, std = float(np.mean(errors)), float(np.std(errors))
    threshold = mean + settings.z_threshold * std
    tier2 = settings.z_threshold_tiers[-1] if len(settings.z_threshold_tiers) > 1 else settings.z_threshold
    threshold_2 = mean + tier2 * std
    return threshold, threshold_2


def predict(scores: np.ndarray, threshold: float, threshold_2: float) -> ThresholdResult:
    pred = np.where(scores >= threshold, "Anomalous", "Normal")
    pred_2 = np.where(scores >= threshold_2, "Anomalous", "Normal")
    return ThresholdResult(
        threshold=threshold,
        threshold_2=threshold_2,
        prediction=pred,
        prediction_2=pred_2,
    )
