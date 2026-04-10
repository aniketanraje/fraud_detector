""" Stateful preprocessing – schema validation, scaling and train/test splitting."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constants import RANDOM_STATE, SCHEMA_CONFIG_PATH
from src.domain.entities import PipelineContext, PhaseResult

from src.domain.exceptions import (
    DataQualityError,
    PreprocessingError,
    SchemaViolationError,
    SplittingError,
)

logger: logging.Logger = logging.getLogger(__name__)

class DataValidator:
    """Validates a Dataframe against the project schema.
    Loads column and type constraints from schema.yaml.
    Fails fast on missing or mis-typed columns before any transformation is applied.

    Attributes:
        schema: Parse schema configuration dictionary.
    """
    def __init__(self) -> None:
        """Load schema configurations from schema.yaml."""
        with open(SCHEMA_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.schema: dict = cfg["schema"]
        self._expected_features: list[str] = self.schema["feature_columns"]
        self._target: str = self.schema["target_column"]
        self._expected_count: int = self.schema["expected_feature_count"]
        self._constraints: dict = self.schema.get("constraints", {})

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the Dataframe against the schema rules.

        Args:
            df: Input Dataframe to validate.

        Returns:
            Validated Dataframe (unchanged if all checks pass).

        Raises:
            SchemaViolationError: If required columns are missing or dtype is wrong.
            DataQualityError: If constraint violation are detected. e.g. Amount < 0.
        """
        # Missing column check
        all_required = self._expected_features + [self._target]
        missing = [c for c in all_required if c not in df.columns]
        if missing:
            raise SchemaViolationError(
                f"Missing required columns: {len(missing)}: {missing}"
            )

        # Extra column warning
        extra = [c for c in df.columns if c not in all_required]
        if extra:
            logger.warning("Extra columns detected (will be ignored): %s", extra)

        # Dtype check – all features must be numeric
        non_numeric = [
            c for c in all_required
            if not pd.api.types.is_numeric_dtype(df[c])
        ]

        if non_numeric:
            raise SchemaViolationError(
                f"Non-numeric columns detected: {non_numeric}"
            )


        # Constraint validation
        if "Amount" in self._constraints:
            min_amount = self._constraints["Amount"].get("min", None)
            if min_amount is not None and (df["Amount"] < min_amount).any():
                n_violations = int((df["Amount"] < min_amount).sum())
                raise DataQualityError(
                    f"Amount constraint violated — {n_violations} rows have Amount < {min_amount}"
                )
        logger.info("Schema validation passed — shape: %s", df.shape)
        return df[all_required].copy()



class Preprocessor:
    """Stateful preprocessing pipeline – scaler fit only on training data.
    Applies StandardScaler to Time and Amount. Preserves V1-V28 unchanged.

    Attributes:
        scaler: Fitted StandardScaler object. None until fit_transform() is called.
        feature_order: Ordered list of column names used during training.
        _scale_cols: Columns passed through StandardScaler.
    """
    def __init__(self) -> None:
        """Initialize Preprocessor with an Unfitted Scaler."""
        with open(SCHEMA_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        schema = cfg["schema"]
        self.scaler: Optional[StandardScaler] = None
        self.feature_order: list[str] = schema["feature_columns"]
        self._scale_cols: list[str] = schema["scaled_features"]
        self._target: str = schema["target_column"]


    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training features and transform them.

        Must only be called on training data – never on the full dataset.

        Args:
            df: Input Dataframe to transform.

        Returns:
            Tuple of (X_scaled,y) numpy arrays.

        Raises:
            PreprocessorError: If scaling or feature extraction fails.
        """
        try:
            X = df[self.feature_order].copy()
            y = df[self._target].values

            self.scaler = StandardScaler()
            X[self._scale_cols] = self.scaler.fit_transform(X[self._scale_cols])

            logger.info("Scaler fitted on %d samples.", X.shape[0])
            return X.values, y
        except Exception as e:
            raise PreprocessingError(f"fit_transform() failed: {e}") from e

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using the already fitted scaler.

        Args:
            df: Dataframe containing features in training order.

        Returns:
            Scaled features matrix as numpy array.

        Raises:
            PreprocessorError: If scaler is not fitted or transform fails.
        """
        if self.scaler is None:
            raise PreprocessingError(
                "Scaler is not fitted. Call fit_transform() on training data first."
            )
        try:
            X = df[self.feature_order].copy()
            X[self._scale_cols] = self.scaler.transform(X[self._scale_cols])
            return X.values
        except Exception as e:
            raise PreprocessingError(f"transform() failed: {e}") from e

class DataSplitter:
    """Splits validated data into stratified train/test sets.

    Stratification preserves the class imbalance ration across splits.

    Attributes:
        test_size: Fraction of data reserved for testing .
        random_state: Random state for reproducibility.
    """

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = RANDOM_STATE,
    ) -> None:
        """Initialize DataSplitter.

               Args:
                   test_size: Proportion of the dataset for the test split.
                   random_state: Random seed.
               """
        self.test_size = test_size
        self.random_state = random_state

    def split(
            self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform stratified train/test split.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            SplittingError: If splitting fails.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )
            logger.info(
                "Split complete — train: %d, test: %d | fraud_train: %d, fraud_test: %d",
                len(y_train),
                len(y_test),
                int(y_train.sum()),
                int(y_test.sum()),
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise SplittingError(f"Train/test split failed: {e}") from e



