"""Phase-contract enforced pipeline orchestrator — strict lifecycle execution."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

from src.constants import (
    MODELS_DIR,
    RANDOM_STATE,
    SCHEMA_CONFIG_PATH,
    TRAIN_CONFIG_PATH,
)
from src.domain.entities import PhaseResult, PhaseTraceEntry, PipelineContext
from src.domain.exceptions import (
    DataIngestionError,
    DataValidationError,
    EvaluationError,
    FeatureEngineeringError,
    ModelRegistrationError,
    PreprocessingError,
    ReportingError,
    SplittingError,
    TrainingError,
)
from src.infrastructure.hf_client import DataIngestor
from src.infrastructure.mlflow_repo import MLflowRepository
from src.infrastructure.storage import ArtifactStore
from src.application.preprocessor import DataSplitter, DataValidator, Preprocessor
from src.application.trainer import ModelTrainer

logger: logging.Logger = logging.getLogger(__name__)


# Phase Protocol

@runtime_checkable
class PhaseProtocol(Protocol):
    """Runtime-checkable interface every pipeline phase must satisfy."""

    def run(self, context: PipelineContext) -> PhaseResult: ...


# Phase Implementation

class IngestionPhase:
    """Downloads and caches the dataset from HuggingFace."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Execute data ingestion.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with raw DataFrame in payload.

        Raises:
            DataIngestionError: If download or caching fails.
        """
        try:
            ingestor = DataIngestor()
            context.raw_df = ingestor.fetch()
            return PhaseResult(
                phase_name="ingestion",
                success=True,
                payload={"shape": context.raw_df.shape},
            )
        except Exception as e:
            raise DataIngestionError(f"Ingestion phase failed: {e}") from e


class ValidationPhase:
    """Validates schema and data quality constraints."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Execute schema and quality validation.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult indicating validation success.

        Raises:
            DataValidationError: If schema or quality checks fail.
        """
        try:
            assert context.raw_df is not None, "raw_df is None — ingestion must run first."
            validator = DataValidator()
            context.validated_df = validator.validate(context.raw_df)
            return PhaseResult(
                phase_name="validation",
                success=True,
                payload={"shape": context.validated_df.shape},
            )
        except (AssertionError, DataValidationError) as e:
            raise DataValidationError(f"Validation phase failed: {e}") from e
        except Exception as e:
            raise DataValidationError(f"Validation phase failed: {e}") from e


class PreprocessingPhase:
    """Scales features and attaches the fitted Preprocessor to context."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Execute preprocessing — scaler fit + transform.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with processed shape in payload.

        Raises:
            PreprocessingError: If scaling fails.
        """
        try:
            assert context.validated_df is not None, "validated_df is None."
            preprocessor = Preprocessor()

            with open(SCHEMA_CONFIG_PATH, "r", encoding="utf-8") as f:
                schema_cfg = yaml.safe_load(f)
            target = schema_cfg["schema"]["target_column"]
            features = schema_cfg["schema"]["feature_columns"]

            context.processed_df = context.validated_df.copy()
            context.config["_preprocessor"] = preprocessor
            context.config["_feature_order"] = features
            context.config["_target"] = target

            return PhaseResult(
                phase_name="preprocessing",
                success=True,
                payload={"features": len(features)},
            )
        except AssertionError as e:
            raise PreprocessingError(f"Preprocessing phase failed: {e}") from e
        except Exception as e:
            raise PreprocessingError(f"Preprocessing phase failed: {e}") from e


class SplittingPhase:
    """Splits the dataset into stratified train/test sets."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Execute stratified train/test split.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with split sizes in payload.

        Raises:
            SplittingError: If splitting fails.
        """
        try:
            assert context.processed_df is not None, "processed_df is None."
            preprocessor: Preprocessor = context.config["_preprocessor"]
            train_cfg = yaml.safe_load(
                open(TRAIN_CONFIG_PATH, "r", encoding="utf-8")
            )["training"]

            splitter = DataSplitter(
                test_size=train_cfg["test_size"],
                random_state=train_cfg["random_state"],
            )

            X_all, y_all = preprocessor.fit_transform(context.processed_df)
            X_train, X_test, y_train, y_test = splitter.split(X_all, y_all)

            context.X_train = X_train
            context.X_test = X_test
            context.y_train = y_train
            context.y_test = y_test
            context.config["_preprocessor"] = preprocessor

            return PhaseResult(
                phase_name="splitting",
                success=True,
                payload={
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "fraud_train": int(y_train.sum()),
                },
            )
        except AssertionError as e:
            raise SplittingError(f"Splitting phase failed: {e}") from e
        except Exception as e:
            raise SplittingError(f"Splitting phase failed: {e}") from e


class TrainingPhase:
    """Trains all configured models."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Execute model training for all configured model names.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with trained model names in payload.

        Raises:
            TrainingError: If any model fails to train.
        """
        try:
            assert context.X_train is not None, "X_train is None — split must run first."
            train_cfg = yaml.safe_load(
                open(TRAIN_CONFIG_PATH, "r", encoding="utf-8")
            )

            model_names = ["xgboost", "random_forest"]

            for name in model_names:
                params = train_cfg.get(name, {})
                trainer = ModelTrainer(
                    model_name=name,
                    params=params,
                    random_state=RANDOM_STATE,
                )
                model = trainer.train(context.X_train, context.y_train)
                context.trained_models[name] = model

            return PhaseResult(
                phase_name="training",
                success=True,
                payload={"models_trained": list(context.trained_models.keys())},
            )
        except AssertionError as e:
            raise TrainingError(f"Training phase failed: {e}") from e
        except Exception as e:
            raise TrainingError(f"Training phase failed: {e}") from e


class EvaluationPhase:
    """Evaluates all trained models and selects the best by AUPRC."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Evaluate all trained models on the test set.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with best model name and AUPRC in payload.

        Raises:
            EvaluationError: If metric computation fails.
        """
        try:
            assert context.trained_models, "trained_models is empty."

            best_auprc = -1.0
            best_name = ""

            for name, model in context.trained_models.items():
                trainer = ModelTrainer(model_name=name)
                metrics = trainer.evaluate(model, context.X_test, context.y_test)
                context.evaluation_results[name] = metrics

                if metrics["auprc"] > best_auprc:
                    best_auprc = metrics["auprc"]
                    best_name = name

            context.best_model_name = best_name
            logger.info(
                "Best model: %s | AUPRC: %.4f", best_name, best_auprc
            )

            return PhaseResult(
                phase_name="evaluation",
                success=True,
                payload={"best_model": best_name, "auprc": best_auprc},
            )
        except AssertionError as e:
            raise EvaluationError(f"Evaluation phase failed: {e}") from e
        except Exception as e:
            raise EvaluationError(f"Evaluation phase failed: {e}") from e


class RegistrationPhase:
    """Persists the best model, scaler, and feature order as versioned artifacts."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Register the best model artifacts.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with version and artifact paths in payload.

        Raises:
            ModelRegistrationError: If artifact saving fails.
        """
        try:
            assert context.best_model_name, "best_model_name is empty."
            store = ArtifactStore(base_dir=MODELS_DIR)
            version, version_dir = store.create_version()
            context.model_version = version

            best_model = context.trained_models[context.best_model_name]
            preprocessor: Preprocessor = context.config["_preprocessor"]

            store.save_pickle(best_model, version_dir, "model.pkl")
            store.save_pickle(preprocessor.scaler, version_dir, "scaler.pkl")
            store.save_json(preprocessor.feature_order, version_dir, "feature_order.json")

            logger.info("Artifacts registered — version: %s", version)

            return PhaseResult(
                phase_name="registration",
                success=True,
                payload={"version": version, "dir": str(version_dir)},
            )
        except AssertionError as e:
            raise ModelRegistrationError(f"Registration phase failed: {e}") from e
        except Exception as e:
            raise ModelRegistrationError(f"Registration phase failed: {e}") from e


class ReportingPhase:
    """Logs the best model run to MLflow."""

    def run(self, context: PipelineContext) -> PhaseResult:
        """Log experiment artifacts and metrics to MLflow.

        Args:
            context: Shared pipeline state.

        Returns:
            PhaseResult with MLflow run ID in payload.

        Raises:
            ReportingError: If MLflow logging fails.
        """
        try:
            base_cfg = yaml.safe_load(
                open("configs/base_config.yaml", "r", encoding="utf-8")
            )
            mlflow_cfg = base_cfg["mlflow"]

            repo = MLflowRepository(
                tracking_uri=mlflow_cfg["tracking_uri"],
                experiment_name=mlflow_cfg["experiment_name"],
                tags=mlflow_cfg["tags"],
            )

            best_model = context.trained_models[context.best_model_name]
            preprocessor: Preprocessor = context.config["_preprocessor"]
            metrics = context.evaluation_results[context.best_model_name]
            y_prob = best_model.predict_proba(context.X_test)[:, 1]

            train_cfg = yaml.safe_load(
                open(TRAIN_CONFIG_PATH, "r", encoding="utf-8")
            )
            params = train_cfg.get(context.best_model_name, {})
            params["model_name"] = context.best_model_name

            run_id = repo.log_run(
                model=best_model,
                scaler=preprocessor.scaler,
                metrics=metrics,
                params=params,
                y_test=context.y_test,
                y_prob=y_prob,
                feature_names=preprocessor.feature_order,
                model_version=context.model_version,
                requirements_path="requirements.txt",
            )

            return PhaseResult(
                phase_name="reporting",
                success=True,
                payload={"mlflow_run_id": run_id},
            )
        except Exception as e:
            raise ReportingError(f"Reporting phase failed: {e}") from e


# ── Orchestrator ───────────────────────────────────────────────────────────────

_PHASES: list[tuple[str, PhaseProtocol]] = [
    ("ingestion", IngestionPhase()),
    ("validation", ValidationPhase()),
    ("preprocessing", PreprocessingPhase()),
    ("splitting", SplittingPhase()),
    ("training", TrainingPhase()),
    ("evaluation", EvaluationPhase()),
    ("registration", RegistrationPhase()),
    ("reporting", ReportingPhase()),
]

_CONTEXT_GUARDS: dict[str, list[str]] = {
    "ingestion":    [],
    "validation":   ["raw_df"],
    "preprocessing":["validated_df"],
    "splitting":    ["processed_df"],
    "training":     ["X_train", "X_test", "y_train", "y_test"],
    "evaluation":   ["trained_models"],
    "registration": ["evaluation_results", "best_model_name"],
    "reporting":    ["model_version"],
}


class FraudPipeline:
    """Strict lifecycle orchestrator for the fraud detection training pipeline.

    Executes phases in fixed order. After each phase, verifies PhaseResult.success
    and validates required context fields. On failure, logs and halts immediately.
    """

    def run(self, context: PipelineContext | None = None) -> PipelineContext:
        """Execute all pipeline phases sequentially.

        Args:
            context: Optional pre-initialised PipelineContext. Creates one if None.

        Returns:
            Completed PipelineContext after all phases succeed.

        Raises:
            The typed exception raised by the failing phase — never swallowed.
        """
        if context is None:
            context = PipelineContext()

        context.run_start = time.time()
        logger.info("═" * 60)
        logger.info("FraudPipeline — starting run")
        logger.info("═" * 60)

        for phase_name, phase_instance in _PHASES:
            assert isinstance(phase_instance, PhaseProtocol), (
                f"{phase_name} does not implement PhaseProtocol"
            )

            logger.info("── Phase: %s", phase_name)
            t0 = time.time()

            result: PhaseResult = phase_instance.run(context)
            duration = round(time.time() - t0, 3)

            if not result.success:
                entry = PhaseTraceEntry(
                    phase_name=phase_name,
                    duration_seconds=duration,
                    success=False,
                    error="PhaseResult.success is False",
                )
                context.trace_log.append(entry)
                logger.error("Phase returned success=False — halting: %s", phase_name)
                self._print_trace(context)
                raise RuntimeError(f"Phase failed (success=False): {phase_name}")

            self._validate_context(context, phase_name)

            entry = PhaseTraceEntry(
                phase_name=phase_name,
                duration_seconds=duration,
                success=True,
            )
            context.trace_log.append(entry)
            logger.info("   ✓ %s — %.3fs | %s", phase_name, duration, result.payload)

        self._print_trace(context)
        return context

    def _validate_context(self, context: PipelineContext, phase_name: str) -> None:
        """Assert required context fields exist after a phase completes.

        Args:
            context: Shared pipeline state.
            phase_name: Name of the phase just completed.

        Raises:
            RuntimeError: If a required context field is missing.
        """
        required_fields = _CONTEXT_GUARDS.get(phase_name, [])
        for field_name in required_fields:
            value = getattr(context, field_name, None)
            is_missing = (
                value is None
                or (hasattr(value, "__len__") and len(value) == 0)
            )
            if is_missing:
                raise RuntimeError(
                    f"Context guard failed after '{phase_name}' — "
                    f"'{field_name}' is missing or empty."
                )

    def _print_trace(self, context: PipelineContext) -> None:
        """Print a formatted phase timing summary to the log.

        Args:
            context: Shared pipeline state containing the trace log.
        """
        total = round(time.time() - context.run_start, 2)
        logger.info("═" * 60)
        logger.info("Pipeline Trace Summary")
        logger.info("═" * 60)

        slowest = max(context.trace_log, key=lambda e: e.duration_seconds, default=None)

        for entry in context.trace_log:
            status = "✓" if entry.success else "✗"
            tag = " ← SLOWEST" if entry is slowest else ""
            logger.info(
                "  %s %-20s %.3fs%s",
                status,
                entry.phase_name,
                entry.duration_seconds,
                tag,
            )

        logger.info("  Total runtime: %.2fs", total)
        logger.info("═" * 60)
