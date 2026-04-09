"""Enums, static paths, and immutable constants for the fraud detection system."""

from __future__ import annotations

import os 
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project Root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Config File Paths
BASE_CONFIG_PATH: Path = CONFIGS_DIR / "base_config.yaml"
TRAIN_CONFIG_PATH: Path = CONFIGS_DIR / "train_config.yaml"
SCHEMA_CONFIG_PATH: Path = CONFIGS_DIR / "schema.yaml"

# Data Source
HF_DATASET_URL: str = os.getenv(
    "HF_DATASET_URL",
    "https://huggingface.co/datasets/sherlockab/creditcard_dataset/resolve/main/creditcard.csv",
)

# Reproducibility
RANDOM_STATE: int = 42

# Pipeline Enums
class PipelineMode(str, Enum):
    TRAIN = "train"
    PREDICT = "predict"
    BATCH = "batch"

class ModelName(str, Enum):
    """Supported model identifiers."""
    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"

class ResamplingStrategy(str, Enum):
    """Supported class imbalance handling strategies."""
    SMOTE = "smote"
    NONE = "none"


# Ensure directories exist at import time
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, LOGS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

