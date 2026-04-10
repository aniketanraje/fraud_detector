"""MLflow tracking and registry wrapper – pro-grade experiment logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import precision_recall_curve

import torch

from src.domain.exceptions import ReportingError


logger: logging.Logger = logging.getLogger(__name__)


class MLflowRepository:
    """Wraps MLflow tracking APIs for structured experiment logging.

    Logs metrics, parameters, artifacts, and tags in a single coherent experiment run.
    Precision-recall curves and feature importance plots are saved as artifact images.

    Attributes:
        tracking_uri: URI for the MLflow tracking server or local repository.
        experiment_name: Name of the MLflow experiment.
        tags: Additional key-value tags applied to every experiment.
    """

    def __init__(
            self,
            tracking_uri: str = "mlruns",
            experiment_name: str = "credit_fraud_detection",
            tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Initializes the MLFflowRepository object.

        Args:
            tracking_uri: URI for the MLflow tracking server or local repository.
            experiment_name: Name of the MLflow experiment.
            tags: Additional key-value tags applied to every experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.tags = tags or {
            "developer": "Aniket Bhosale",
            "env": "production",
            "framework": "scikit-learn",
        }

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow initialized — experiment: %s", self.experiment_name)

    def log_run(
            self,
            model: Any,
            scaler: Any,
            metrics: dict[str, float],
            params: dict[str, Any],
            y_test: np.ndarray,
            y_prob: np.ndarray,
            feature_names: list[str],
            model_version: str,
            requirements_path: Optional[Path] = None,
    ) -> str:
        """Log a complete training run to MLflow.

        Logs parameters, metrics, model artifacts, scaler artifact, plots,
        and requirements as a conda environment file.

        Args:
            model: Fitted sklearn-compatible model.
            scaler: Fitted StandardScaler.
            metrics: Dictionary of metrics associated with each metric.
            params: Dictionary of model parameters.
            y_test: Array of true labels.
            y_prob: Array of predicted labels.
            feature_names: Ordered list of feature names.
            model_version: Version string (e.g., 'v_1720000000').
            requirements_path: Optional path to requirements.txt for conda logging.

        Returns:
            The MLflow run ID.

        Raises:
            ReportingError: If MLflow tracking server or local repository fails.
        """
        try:
            with mlflow.start_run(tags=self.tags) as run:
                # parameters
                mlflow.log_params(params)
                mlflow.log_param("model_type", params.get("model_name"))

                if params.get("model_name") == "pytorch_mlp":
                    mlflow.log_param("framework", "pytorch")
                    mlflow.log_param("epochs", getattr(model, "epochs", None))
                mlflow.log_param("model_version", model_version)

                # Metrics - prioritise AUPRC
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)

                # Model artifacts
                # Model artifacts
                if params.get("model_name") == "pytorch_mlp":
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        torch.save(model.model.state_dict(), tmp.name)
                        tmp_path = tmp.name

                    mlflow.log_artifact(tmp_path, artifact_path="model")
                    os.unlink(tmp_path)

                else:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name="fraud_detector"
                    )

                # Scaler as generic artifact
                import tempfile, pickle
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    pickle.dump(scaler, tmp)
                    tmp_path = tmp.name
                mlflow.log_artifact(tmp_path, artifact_path="scaler")
                os.unlink(tmp_path)

                # Feature Order
                import tempfile, json
                with tempfile.NamedTemporaryFile(
                        suffix=".json", mode="w", delete=False, encoding="utf-8"
                ) as tmp:
                    json.dump(feature_names, tmp)
                    tmp_path = tmp.name
                mlflow.log_artifact(tmp_path, artifact_path="schema")
                os.unlink(tmp_path)

                # Precision-Recall curve plot

                pr_fig = self._plot_precision_recall_curve(y_test, y_prob)
                mlflow.log_figure(pr_fig, "plots/precision_recall_curve.png")
                plt.close(pr_fig)

                # Feature Importance Plot (if available)
                fi_fig = self._plot_feature_importance(model, feature_names)
                if fi_fig is not None:
                    mlflow.log_figure(fi_fig, "plots/feature_importance.png")
                    plt.close(fi_fig)

                # Requirements as conda env artifact
                if requirements_path and Path(requirements_path).exists():
                    mlflow.log_artifact(str(requirements_path), artifact_path="environment")

                run_id = run.info.run_id
                logger.info("MLflow run logged — run_id: %s", run_id)
                return run_id

        except Exception as e:
            raise ReportingError(f"MLflow logging failed: {e}") from e

    def _plot_precision_recall_curve(
            self,
            y_test: np.ndarray,
            y_prob: np.ndarray,
    ) -> plt.Figure:
        """Generate a Precision-Recall curve plot.

        Args:
            y_test: Array of true labels.
            y_prob: Array of predicted labels.

        Returns:
            Matplotlib figure object.
        """

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="darkorange", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_feature_importance(
            self,
            model: Any,
            feature_names: list[str],
            top_n: int = 20,
    ) -> Optional[plt.Figure]:
        """Generate a feature importance plot if the model supports it.

        Args:
            model: Fitted model with optional feature_importances_ attribute.
            feature_names: Ordered feature names.
            top_n: Number of top features to display.

        Returns:
            Matplotlib figure object.
        """
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            coef = getattr(model, "coef_", None)
            if coef is not None:
                importances = np.abs(coef).flatten()
            else:
                return None

        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_importances[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        return fig
