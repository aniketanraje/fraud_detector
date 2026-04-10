"""PyTorch-based MLP model with sklearn-compatible interface for fraud detection."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger: logging.Logger = logging.getLogger(__name__)


class FraudMLP(nn.Module):
    """Feedforward neural network for tabular fraud detection."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchModelWrapper:
    """Sklearn-compatible wrapper around a PyTorch MLP model.

    Handles training, inference, and probability prediction in a unified interface.
    """

    def __init__(
        self,
        input_dim: int,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        self.input_dim: int = input_dim
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.lr: float = lr
        self.weight_decay: float = weight_decay

        self.device: torch.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: FraudMLP = FraudMLP(input_dim).to(self.device)
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchModelWrapper:
        """Train the neural network on tabular data."""

        try:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            # Handle class imbalance
            pos_weight = torch.tensor(
                [(len(y) - y.sum()) / (y.sum() + 1e-8)],
                dtype=torch.float32,
                device=self.device,
            )

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            self.model.train()

            for epoch in range(self.epochs):
                epoch_loss: float = 0.0

                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)
                logger.info(
                    "TorchMLP — epoch %d/%d | loss: %.6f",
                    epoch + 1,
                    self.epochs,
                    avg_loss,
                )

            self._is_fitted = True
            return self

        except Exception as e:
            raise RuntimeError(f"TorchModelWrapper.fit() failed: {e}") from e

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities in sklearn-compatible format."""

        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        try:
            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()

            return np.hstack([1 - probs, probs])

        except Exception as e:
            raise RuntimeError(f"predict_proba() failed: {e}") from e