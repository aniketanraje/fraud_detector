"""Project-specific exceptions hierarchy – typed errors per pipeline phase."""

from __future__ import annotations

class ProjectError(Exception):
    """Base exception for all errors raised by this project."""
    pass


class ConfigurationError(ProjectError):
    """Raise when  configuration loading or validatiion fails."""
    pass


class DataIngestionError(ProjectError):
    """Raised when data download, caching, or 10 operations fail."""
    pass


class DataValidationError(ProjectError):
    """Raised when the loaded dataset fails schema or integrity checks."""
    pass


class SchemaValidationError(DataValidationError):
    """Raised when the loaded dataset fails schema or integrity checks."""
    pass


class DataQualityError(DataValidationError):
    """Raised when data quality constraints are violated. (e.g. Amount < 0 )."""
    pass


class PreprocessingError(ProjectError):
    """Raised when scaling, imputation, or transformation operations fail"""
    pass


class FeatureEngineeringError(ProjectError):
    """Raised when feature construction or selection fails."""
    pass


class SplittingError(ProjectError):
    """Raised when train/test splitting fails."""
    pass


class ModelRegistrationError(ProjectError):
    """Raised when model artifact persistence or versioning fails."""
    pass



class TrainingError(ProjectError):
    """Raised when model fitting or cross-validation fails."""
    pass


class ConvergenceError(TrainingError):
    """Raised when a model fails to converge during training."""
    pass


class EvaluationError(TrainingError):
    """Raised when metric computation or evaluation reporting fails."""
    pass


class ReportingError(ProjectError):
    """Raise when MLflow logging or artifact export fails."""
    pass


class PredictionError(ProjectError):
    """Raised when inference fails due to model or input issues."""
    pass