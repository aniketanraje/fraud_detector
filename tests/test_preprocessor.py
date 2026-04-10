"""Unit tests for preprocessing — DataValidator, Preprocessor, DataSplitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.preprocessor import DataValidator, Preprocessor, DataSplitter
from src.domain.exceptions import (
    DataQualityError,
    PreprocessingError,
    SchemaViolationError,
    SplittingError,
)


# Fixtures

@pytest.fixture
def valid_df() -> pd.DataFrame:
    data = {
        "Time": [0.0, 1.0, 2.0],
        "Amount": [100.0, 200.0, 300.0],
        "Class": [0, 1, 0],
        **{f"V{i}": [0.0, 0.1, 0.2] for i in range(1, 29)},
    }
    return pd.DataFrame(data)


# DataValidator

class TestDataValidator:

    def test_valid_dataframe_passes(self, valid_df):
        validator = DataValidator()
        out = validator.validate(valid_df)
        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == 3

    def test_missing_column_raises(self, valid_df):
        df = valid_df.drop(columns=["V1"])
        validator = DataValidator()

        with pytest.raises(SchemaViolationError, match="Missing required columns"):
            validator.validate(df)

    def test_non_numeric_column_raises(self, valid_df):
        df = valid_df.copy()
        df["V2"] = ["a", "b", "c"]

        validator = DataValidator()
        with pytest.raises(SchemaViolationError, match="Non-numeric columns"):
            validator.validate(df)

    def test_amount_constraint_violation(self, valid_df, monkeypatch):
        df = valid_df.copy()
        df.loc[0, "Amount"] = -100

        validator = DataValidator()

        # inject constraint dynamically
        validator._constraints = {"Amount": {"min": 0}}

        with pytest.raises(DataQualityError, match="Amount constraint violated"):
            validator.validate(df)


# Preprocessor

class TestPreprocessor:

    def test_fit_transform_returns_numpy(self, valid_df):
        pre = Preprocessor()
        X, y = pre.fit_transform(valid_df)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

    def test_scaler_is_fitted(self, valid_df):
        pre = Preprocessor()
        pre.fit_transform(valid_df)

        assert pre.scaler is not None

    def test_transform_without_fit_raises(self, valid_df):
        pre = Preprocessor()

        with pytest.raises(PreprocessingError, match="Scaler is not fitted"):
            pre.transform(valid_df)

    def test_transform_after_fit(self, valid_df):
        pre = Preprocessor()
        pre.fit_transform(valid_df)

        X_transformed = pre.transform(valid_df)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(valid_df)

    def test_scaling_changes_distribution(self, valid_df):
        pre = Preprocessor()
        X_scaled, _ = pre.fit_transform(valid_df)

        # scaled columns should have mean approx 0
        assert np.allclose(X_scaled.mean(axis=0)[0:2], 0, atol=1)


# DataSplitter

class TestDataSplitter:

    def test_split_shapes(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        splitter = DataSplitter(test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        assert len(X_train) + len(X_test) == 100
        assert len(y_train) + len(y_test) == 100

    def test_stratification_preserved(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        splitter = DataSplitter(test_size=0.2)
        _, _, y_train, y_test = splitter.split(X, y)

        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)

        assert abs(train_ratio - test_ratio) < 0.05

    def test_split_failure_raises(self):
        splitter = DataSplitter()

        with pytest.raises(SplittingError):
            splitter.split(np.array([]), np.array([]))