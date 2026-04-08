"""Pydantic data models – TransactionInput, BatchSchema, PhaseResult, PipelineContext"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from mlflow.entities.model_registry import model_version

from pydantic import BaseModel, field_validator, model_validator

# Inference Models

class TransactionInput(BaseModel):
    """Pydantic schema for a single transaction inference request.
    Validates all 30 feature columns. Fails fast on missing fields or
    constraint violations before the input reaches the Predictor.
    """
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    @field_validator("Amount")
    @classmethod
    def amount_must_be_non_negative(cls, v:float) -> float:
        """Validate tat Amount is non-negative.
        Args:
            v (float): Amount to be validated.

        Returns:
            float: Amount validated.

        Raises:
            ValueError: Amount is not non-negative.
        """
        if v < 0:
            raise ValueError(f"Amount must be >=0 i.e. non-negative, got {v}")
        return v

    @field_validator("Time")
    @classmethod
    def time_must_be_non_negative(cls, v:float) -> float:
        """Validate tat Time is non-negative.

        Args:
            v (float): Time to be validated.

        Returns:
            float: Time to be validated.

        Raises:
            ValueError: Time is not non-negative.
        """
        if v < 0:
            raise ValueError(f"Time must be >=0 i.e. non-negative, got {v}")
        return v

class PredictionOutput(BaseModel):
    """Structured output from the fraud prediction service.

    Attributes:
        is_fraud: Boolean fraud classification.
        probability: Fraud probability score in [0, 1].
        model_version: Identifier of the model version used.
        risk_level: Human-readable risk tier.
    """

    is_fraud: bool
    probability: float
    model_version: str
    risk_level: str

    @field_validator("probability")
    @classmethod
    def probability_in_range(cls, v: float) -> float:
        """Validate probability is between 0 and 1.
        Args:
            v (float): Probability to be validated.

        Returns:
            float: Probability validated.

        Raises:
            ValueError: Probability is not [0, 1].
        """

        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Probability must be [0, 1], got {v}")
        return v


# Pipeline orchestration models

@dataclass
class PhaseResult:
    """Uniform result container returned by every pipeline phase.

    Attributes:
        phase_name: Identifier of the phase that produced this result.
        success: Whether the phase completed without errors.
        payload: Arbitrary metadata or artifacts produced by this phase.
    """

    phase_name: str
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass
class PhaseTraceEntry:
    """Timing and status trace for a single pipeline phase.
    Attributes:
        phase_name: Phase Identifier.
        duration_seconds: Wall-clock execution time in seconds.
        success: Whether the phase completed without errors.
        error: Error message if phase failed, else None.
    """

    phase_name: str
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineContext:
    """Shared mutable state passed through every pipeline phase.

    All intermediate DataFrames, splits, models, and evaluation results
    are attached here. No phase writes to any other global state.

    Attributes:
        raw_df: Dataframe after ingestion.
        validated_df: Dataframe after schema validation.
        processed_df: Dataframe after processing phase.
        feature_df: Dataframe after feature engineering phase.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training label matrix.
        y_test: Test label matrix.
        trained_models: Dictionary of fitted model objects keyed by name. 
        evaluation_results: Dictionary of metric dictionaries keyed by model name.
        best_model_name: Name of the best performing model.
        model_version: Version string for the registered artifact. 
        trace_log: Ordered list of phase execution trace.
        config: Full merged configuration dictionary.
        run_start: Pipeline start timestamp.
    """
    raw_df: Optional[pd.DataFrame] = None
    validated_df: Optional[pd.DataFrame] = None
    processed_df: Optional[pd.DataFrame] = None
    feature_df: Optional[pd.DataFrame] = None
    X_train: Optional[Any] = None
    X_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None
    trained_models: dict[str, Any] = field(default_factory=dict)
    evaluation_results: dict[str, dict[str, float]] = field(default_factory=dict)
    best_model_name: str = ""
    model_version: str = ""
    trace_log: list[PhaseTraceEntry] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    run_start: float = field(default_factory=time.time)
