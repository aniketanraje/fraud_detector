"""Experiment training logic – XGBoost / sklearn with evaluation metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import RANDOM_STATE
from src.domain.exceptions import EvaluationError, TrainingError

logger: logging.Logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available – failing back to Random Forest")


class ModelTrainer:
    """Trains and evaluates fraud detection models.

    Supports XGBoost(preferred), LogisticRegression, and RandomForest.
    Returns fitted model and evaluation metrics dictionary.

    Attributes:
        model_name: Identifier of the model to train.
        params: Hyperparameters dictionary for the selected model.
        random_state: Global random seed.
    """
    def __init__(
            self,
            model_name: str="xgboost",
            params: dict[str, Any] | None = None,
            random_state: int = RANDOM_STATE,
    ) -> None:
        self.model_name = model_name
        self.params = params or {}
        self.random_state = random_state

    def _build_model(self, y_train: np.ndarray) -> Any:
        """Instantiate the selected model with configured hyperparameters.

        Args:
            y_train: Training labels (used to compute scale_pos_weight for XGBoost).

        Returns:
            Unfitted sklearn-compatible model.

        Raises:
            TrainingError: If the model name is not recognized.
        """
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1

        if self.model_name == "xgboost":
            if not _XGBOOST_AVAILABLE:
                logger.warning("XGBoost is not installed – using Random Forest")
                return self._build_random_forest()

            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "use_label_encoder": False,
                "random_state": self.random_state,
                "n_jobs": -1,
            }

            default_params.update(self.params)
            return XGBClassifier(**default_params)

        elif self.model_name == "logistic_regression":
            default_params = {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": self.random_state,
            }
            default_params.update(self.params)
            return LogisticRegression(**default_params)

        elif self.model_name == "random_forest":
            return self._build_random_forest()
        elif self.model_name == "pytorch_mlp":
            from src.application.dl_model import TorchModelWrapper

            input_dim = self.params.get("input_dim")

            if input_dim is None:
                raise ValueError("input_dim must be provided for pytorch_mlp")

            return TorchModelWrapper(
                input_dim=input_dim,
                epochs=self.params.get("epochs", 10),
                batch_size=self.params.get("batch_size", 1024),
                lr=self.params.get("lr", 1e-3),
                weight_decay=self.params.get("weight_decay", 1e-5),
            )
        else:
            raise TrainingError(f"Unknown model name: {self.model_name}")


    def _build_random_forest(self) -> RandomForestClassifier:
        """Build a RandomForestClassifier with balanced class weights.

        Returns:
            Configured RandomForestClassifier.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        default_params.update(self.params)
        return RandomForestClassifier(**default_params)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
    ) -> Any:
       """Fit the model on training data.

       Args:
           X_train: Training feature matrix.
           y_train: Training labels.

        Returns:
            Fitted model object.

        Raises:
            TrainingError: If fitting fails.
       """
       if self.model_name == "pytorch_mlp":
           self.params["input_dim"] = X_train.shape[1]
       model = self._build_model(y_train)
       logger.info("Training %s on %d samples...", self.model_name, len(y_train))

       try:
           model.fit(X_train, y_train)
           logger.info("Training complete — %s", self.model_name)
           return model
       except Exception as e:
           raise TrainingError(f"{self.model_name} training failed: {e}`") from e

    def evaluate(
            self,
            model: Any,
            X_test: np.ndarray,
            y_test: np.ndarray,
            threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute evaluation metrics on the test set

        Args:
            model: fitted model object.
            X_test: Test feature matrix.
            y_test: True test labels.
            threshold: Decision threshold for binary classification.

        Returns:
            Dictionary of metric name → float value.

        Raises:
            TrainingError: If metric computation fails.
        """
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            metrics = {
                "auprc": float(average_precision_score(y_test, y_prob)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }

            logger.info(
                "%s metrics — AUPRC: %.4f | Recall: %.4f | F1: %.4f",
                self.model_name,
                metrics["auprc"],
                metrics["recall"],
                metrics["f1"],
            )

            return metrics
        except Exception as e:
            raise TrainingError(f"{self.model_name} training failed: {e}") from e
