"""CLI entrypoint - routes train / predict / batch modes to pipeline and predictor"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.constants import LOGS_DIR, PipelineMode
from src.domain.entities import PipelineContext, TransactionInput
from src.domain.exceptions import ProjectError

# Logging Setup

def _configure_logging() -> None:
    """Configure rotating file handler and stream handler for the root logger."""
    log_path = LOGS_DIR / "fraud_detection.log"
    formater = logging.formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formater)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formater)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(stream_handler)

logger: logging.Logger = logging.getLogger(__name__)

# Mode Handlers


def _handle_train() -> None:
    """Execute the full training pipeline.

    Raises:
        ProjectError: If any pipeline phase fails.
    """
    from src.application.pipeline import FraudPipeline

    logger.info("Mode: TRAIN")
    pipeline = FraudPipeline()
    context = PipelineContext()
    completed_context = pipeline.run(context)

    logger.info(
        "Training complete — best model: %s | version: %s",
        completed_context.best_model_name,
        completed_context.model_version,
    )

    best_metrics = completed_context.evaluation_results.get(
        completed_context.best_model_name, {}
    )
    for metric_name, value in best_metrics.items():
        logger.info("  %s: %.4f", metric_name, value)


def _handle_predict(input_json: str) -> None:
    """Run single-transaction inference from a JSON string.

    Args:
        input_json: JSON string representing a transaction.

    Raises:
        ProjectError: If validation or inference fails.
        ValueError: If the JSON cannot be parsed.
    """
    from src.application.predictor import Predictor

    logger.info("Mode: PREDICT")

    try:
        raw = json.loads(input_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}") from e

    transaction = TransactionInput(**raw)
    predictor = Predictor.get_instance()
    result = predictor.predict(transaction)

    output = result.model_dump()
    print(json.dumps(output, indent=2))
    logger.info("Prediction output: %s", output)


def _handle_batch(input_csv: str, output_csv: str) -> None:
    """Run batch inference on a CSV file.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path to write the results CSV.

    Raises:
        ProjectError: If batch inference or IO fails.
        FileNotFoundError: If the input CSV does not exist.
    """
    import pandas as pd
    from src.application.predictor import Predictor

    logger.info("Mode: BATCH | input: %s", input_csv)

    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    predictor = Predictor.get_instance()
    results = predictor.predict_batch(df)
    results.to_csv(output_csv, index=False)

    logger.info(
        "Batch complete — %d rows | output: %s",
        len(results),
        output_csv,
    )
    fraud_count = int(results["is_fraud"].sum())
    logger.info(
        "Fraud flagged: %d / %d (%.2f%%)",
        fraud_count,
        len(results),
        100.0 * fraud_count / len(results),
    )

# CLI

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — ML System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode train
  python -m src.main --mode predict --input '{"Time":0,"Amount":149.62,"V1":-1.36,...}'
  python -m src.main --mode batch --input data/raw/creditcard.csv --output results.csv
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in PipelineMode],
        required=True,
        help="Execution mode: train | predict | batch",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="JSON string (predict mode) or CSV path (batch mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV path for batch mode",
    )
    return parser


def main() -> None:
    """Main entry point — parse CLI arguments and delegate to mode handlers.

    Raises:
        SystemExit: On argument errors or unrecoverable failures.
    """
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    logger.info("Credit Card Fraud Detection — starting | mode: %s", args.mode)

    try:
        if args.mode == PipelineMode.TRAIN.value:
            _handle_train()

        elif args.mode == PipelineMode.PREDICT.value:
            if not args.input:
                parser.error("--input is required for predict mode.")
            _handle_predict(args.input)

        elif args.mode == PipelineMode.BATCH.value:
            if not args.input:
                parser.error("--input is required for batch mode.")
            _handle_batch(args.input, args.output)

    except ProjectError as e:
        logger.error("Pipeline error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


