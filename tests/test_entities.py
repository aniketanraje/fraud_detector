"""Unit tests for Pydantic domain entities — TransactionInput, PredictionOutput."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.entities import PredictionOutput, TransactionInput


# Fixtures

@pytest.fixture
def valid_transaction() -> dict:
    return {
        "Time": 0.0,
        "Amount": 149.62,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }


# TransactionInput

class TestTransactionInput:

    def test_valid_transaction_parses(self, valid_transaction) -> None:
        tx = TransactionInput(**valid_transaction)
        assert tx.Amount == 149.62
        assert tx.Time == 0.0

    def test_negative_amount_raises(self, valid_transaction) -> None:
        bad = {**valid_transaction, "Amount": -1.0}
        with pytest.raises(ValidationError, match="Amount must be >=0"):
            TransactionInput(**bad)

    def test_negative_time_raises(self, valid_transaction) -> None:
        bad = {**valid_transaction, "Time": -5.0}
        with pytest.raises(ValidationError, match="Time must be >=0"):
            TransactionInput(**bad)

    def test_missing_field_raises(self, valid_transaction) -> None:
        bad = {k: v for k, v in valid_transaction.items() if k != "V14"}
        with pytest.raises(ValidationError):
            TransactionInput(**bad)

    def test_zero_amount_is_valid(self, valid_transaction) -> None:
        tx = TransactionInput(**{**valid_transaction, "Amount": 0.0})
        assert tx.Amount == 0.0

    def test_feature_count(self, valid_transaction) -> None:
        tx = TransactionInput(**valid_transaction)
        assert len(tx.model_dump()) == 30

    def test_invalid_type_raises(self, valid_transaction) -> None:
        bad = {**valid_transaction, "Amount": "invalid"}
        with pytest.raises(ValidationError):
            TransactionInput(**bad)


# PredictionOutput

class TestPredictionOutput:

    def test_valid_output_parses(self) -> None:
        out = PredictionOutput(
            is_fraud=True,
            probability=0.92,
            model_version="v_1",
            risk_level="CRITICAL",
        )
        assert out.is_fraud is True
        assert out.probability == 0.92

    def test_probability_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="Probability must be \\[0, 1\\]"):
            PredictionOutput(
                is_fraud=True,
                probability=1.5,
                model_version="v_1",
                risk_level="HIGH",
            )

    def test_probability_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="Probability must be \\[0, 1\\]"):
            PredictionOutput(
                is_fraud=False,
                probability=-0.1,
                model_version="v_1",
                risk_level="LOW",
            )

    def test_probability_boundaries(self) -> None:
        low = PredictionOutput(
            is_fraud=False,
            probability=0.0,
            model_version="v_1",
            risk_level="LOW",
        )
        high = PredictionOutput(
            is_fraud=True,
            probability=1.0,
            model_version="v_1",
            risk_level="CRITICAL",
        )
        assert low.probability == 0.0
        assert high.probability == 1.0

    def test_invalid_types_raise(self) -> None:
        with pytest.raises(ValidationError):
            PredictionOutput(
                is_fraud="yes",
                probability="high",
                model_version=123,
                risk_level=999,
            )