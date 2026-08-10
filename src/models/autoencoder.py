"""
Autoencoder for anomaly detection via reconstruction error.

Uses PyTorch when available; falls back to sklearn MLPRegressor otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from config.settings import settings

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.neural_network import MLPRegressor


class AutoencoderModel(Protocol):
    def predict_reconstruction(self, X: np.ndarray) -> np.ndarray: ...


class SklearnAutoencoder:
    """MLPRegressor trained to reconstruct its own input."""

    backend = "sklearn"

    def __init__(self, model: MLPRegressor, input_dim: int, encoding_dim: int) -> None:
        self.model = model
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def predict_reconstruction(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


if _TORCH_AVAILABLE:

    class TorchAutoencoder(nn.Module):
        backend = "torch"

        def __init__(self, input_dim: int, encoding_dim: int) -> None:
            super().__init__()
            hidden = max(encoding_dim * 2, 64)
            self.input_dim = input_dim
            self.encoding_dim = encoding_dim
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, encoding_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

        def predict_reconstruction(self, X: np.ndarray) -> np.ndarray:
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                t = torch.FloatTensor(X).to(device)
                return self(t).cpu().numpy()


def train_autoencoder(X_train: np.ndarray) -> tuple[Any, list[float]]:
    if _TORCH_AVAILABLE:
        return _train_torch(X_train)
    return _train_sklearn(X_train)


def _train_sklearn(X_train: np.ndarray) -> tuple[SklearnAutoencoder, list[float]]:
    hidden = (max(settings.ae_encoding_dim * 2, 64), settings.ae_encoding_dim)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        max_iter=settings.ae_epochs,
        learning_rate_init=settings.ae_learning_rate,
        random_state=settings.random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, X_train)
    losses = [float(model.loss_) if model.loss_ is not None else 0.0]
    wrapper = SklearnAutoencoder(model, X_train.shape[1], settings.ae_encoding_dim)
    return wrapper, losses


def _train_torch(X_train: np.ndarray) -> tuple[TorchAutoencoder, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_val = max(1, int(len(X_train) * 0.1))
    X_tr = X_train[:-n_val]

    model = TorchAutoencoder(X_train.shape[1], settings.ae_encoding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.ae_learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr)),
        batch_size=min(settings.ae_batch_size, len(X_tr)),
        shuffle=True,
    )
    losses: list[float] = []
    for _ in range(settings.ae_epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / max(len(train_loader), 1))

    return model, losses


def reconstruction_errors(model: Any, X: np.ndarray) -> np.ndarray:
    recon = model.predict_reconstruction(X)
    return np.mean((X - recon) ** 2, axis=1)


def save_model(model: Any, losses: list[float], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    backend = getattr(model, "backend", "sklearn")
    meta = {
        "backend": backend,
        "input_dim": getattr(model, "input_dim", None),
        "encoding_dim": settings.ae_encoding_dim,
        "losses": losses,
    }
    if backend == "torch":
        meta["input_dim"] = model.input_dim
        torch.save(model.state_dict(), path / "model.pt")
    else:
        meta["input_dim"] = model.input_dim
        joblib.dump(model, path / "model.joblib")
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_model(path: Path) -> Any:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    backend = meta.get("backend", "sklearn")
    if backend == "torch" and _TORCH_AVAILABLE:
        model = TorchAutoencoder(meta["input_dim"], meta["encoding_dim"])
        model.load_state_dict(
            torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        return model
    return joblib.load(path / "model.joblib")
