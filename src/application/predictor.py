"""Thread-safe Singleton inference service – loads model and scaler from versioned artifacts."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.constants import MODELS_DIR, SCHEMA_CONFIG_PATH
from src.domain.entities import PredictionOutput, TransactionInput
from src.domain.exceptions import ModelRegistrationError, PredictionError
from src.infrastructure.storage import ArtifactStore

import yaml


logger: logging.Logger = logging.getLogger(__name__)

_HIGH_RISK_THRESHOLD: float = 0.8
_FRAUD_THRESHOLD: float = 0.5

def _resolve_risk_level(probability: float) -> str:
    """Map a fraud probability to a risk level tier which can be understood by humans

    Args:
        probability: Fraud probability in [0, 1]

    Returns:
        risk_level: Risk level. One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    """
    if probability >= _HIGH_RISK_THRESHOLD:
        return "CRITICAL"
    elif probability >= _FRAUD_THRESHOLD:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    return "LOW"


class Predictor:
    """Thread-safe Singleton fraud prediction service.

    Loads the model, scaler, and feature order from the latest versioned
    artifact directory. Feature alignment is enforced at inference time –
    inputs are rendered to match the training vector exactly.

    Usage:
        predictor = Predictor.get_instance()
        result = predictor.predict(transaction_input)
    """
    _instance: Optional["Predictor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private – use Predictor.get_instance() instead."""
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feature_order: list[str] = list()
        self._scale_cols: list[str] = list()
        self._model_version: str =""
        self._store: ArtifactStore = ArtifactStore(base_dir=MODELS_DIR)
        self._loaded: bool = False


    @classmethod
    def get_instance(cls) -> Predictor:
        """Return the global predictor singleton, creating it if necessary.

        Returns:
            The singleton predictor instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._load()
        return cls._instance


    @classmethod
    def reset(cls) -> None:
        """Reset the singleton – used in tests and when reloading a new model version.

        After reset, the next call to get_instance() will load fresh artifacts.
        """
        with cls._lock:
            cls._instance = None
        logger.info(f"Reset singleton predictor: {cls._instance}")

    def _load(self) -> None:
        """Load model, scaler, and feature order from the latest version directory.

        Raises:
            ModelRegistrationError: If no version directory exists or artifacts are missing.
        """
        version_dir = self._store.get_latest_version_dir()
        if version_dir is None:
            raise ModelRegistrationError(
                "No trained model found. Run: python -m src.main --mode train"
            )

        model_path = version_dir / "model.pkl"
        scaler_path = version_dir / "scaler.pkl"
        feature_order_path = version_dir / "feature_order.json"

        for path in (model_path, scaler_path, feature_order_path):
            if not self._store.is_artifact_healthy(path):
                raise ModelRegistrationError(f"Artifact missing or empty: {path}")
        self._model = self._store.load_pickle(model_path)
        self._scaler = self._store.load_pickle(scaler_path)
        self._feature_order = self._store.load_json(feature_order_path)
        self._model_version = version_dir.name

        with open(SCHEMA_CONFIG_PATH, "r", encoding="utf-8") as f:
            schema_cfg = yaml.safe_load(f)
        self._scale_cols = schema_cfg["schema"]["scaled_features"]

        self._loaded = True

        logger.info(
            "Predictor loaded — version: %s | features: %d",
            self._model_version,
            len(self._feature_order),
        )

    def _align_features(self, transaction: TransactionInput) -> np.ndarray:
        """Build a feature vector aligned to the training feature order.

        Args:
            transaction: Validated TransactionInput Pydantic model.

        Returns:
            2D numpy array of shape (1, n_features)

        Raises:
            PredictionError: If feature alignment fails.
        """
        try:
            raw_dict = transaction.model_dump()
            row = {col: raw_dict[col] for col in self._feature_order}
            df = pd.DataFrame([row])
            df[self._scale_cols] = self._scaler.transform(df[self._scale_cols])
            return df[self._feature_order].values
        except Exception as e:
            raise PredictionError(f"Feature alignment fail: {e}") from e

    def predict(self, transaction: TransactionInput) -> PredictionOutput:
        """Run inference on a single validated transaction.

        Args:
            transaction: Validated TransactionInput Pydantic model.

        Returns:
            PredictionOutput with is_fraud, probability, model_version, risk_level.

        Raises:
            PredictionError: If Inference fails for any reason.
        """
        if not self._loaded:
            raise PredictionError("Predictor not loaded. Call Predictor.get_instance().")

        try:
            X = self._align_features(transaction)
            probability = float(self._model.predict_proba(X)[0, 1])
            is_fraud = probability >= _FRAUD_THRESHOLD
            risk_level = _resolve_risk_level(probability)

            logger.info(
                "Prediction — probability: %.4f | is_fraud: %s | risk: %s",
                probability,
                is_fraud,
                risk_level,
            )
            return PredictionOutput(
                is_fraud=is_fraud,
                probability=round(probability, 6),
                model_version=self._model_version,
                risk_level=risk_level,
            )
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Inference failed: {e}") from e

    def predict_batch(self,df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a validated batch DataFrame.

        Args:
            df: Dataframe containing all feature columns in any order.

        Returns:
            Input DataFrame with 'fraud_probability', 'is_fraud', 'risk_level' columns appended.

        Raises:
            PredictionError: If batch inference fails.
        """
        if not self._loaded:
            raise PredictionError("Predictor not loaded.")

        try:
            X = df[self._feature_order].copy()
            X[self._scale_cols] = self._scaler.transform(X[self._scale_cols])
            probabilities = self._model.predict_proba(X.values)[:, 1]

            result = df.copy()
            result["fraud_probability"] = probabilities
            result["is_fraud"] = (probabilities >= _FRAUD_THRESHOLD).astype(int)
            result["risk_level"] = [_resolve_risk_level(p) for p in probabilities]
            result["model_version"] = self._model_version

            fraud_count = int(result["is_fraud"].sum())
            logger.info(
                "Batch prediction complete — %d rows | %d flagged as fraud",
                len(result),
                fraud_count,
            )
            return result

        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Feature alignment fail: {e}") from e


    @property
    def model_version(self) -> str:
        """Return the currently loaded model version string."""
        return self._model_version

    @property
    def is_healthy(self) -> bool:
        """Return True if the model and scaler are loaded and ready"""
        return (
            self._loaded
            and self._model is not None
            and self._scaler is not None
            and len(self._feature_order) > 0
        )