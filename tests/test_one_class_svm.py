import numpy as np

from src.models.one_class_svm import anomaly_scores, train_ocsvm


def test_ocsvm_trains_and_returns_scores() -> None:
    rng = np.random.RandomState(42)
    X = rng.normal(size=(80, 8))

    model = train_ocsvm(X)
    scores = anomaly_scores(model, X)

    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()
    assert np.ptp(scores) >= 0
