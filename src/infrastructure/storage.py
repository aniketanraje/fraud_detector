"""Artifact versioning and IO – immutable versioned model directories."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

from src.constants import MODELS_DIR
from src.domain.exceptions import ModelRegistrationError

logger: logging.Logger = logging.getLogger(__name__)

class ArtifactStore:
    """Manage versioned model artifact persistence.

    Each training run creates a new timestamped directory under models/.
    Artifacts are never overwritten – every version is retained.

    Attributes:
        base_dir: Root directory for all versioned model artifacts.
    """

    def __init__(self, base_dir: Path = MODELS_DIR) -> None:
        """Initialize the ArtifactStore.

        Args:
            base_dir: Root directory for all versioned model artifacts.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)


    def create_version(self) -> tuple[str, Path]:
        """Create a new timestamped version directory.

        Returns:
            Tuple[str, Path]: Tuple of (version_string, version_path).

        Raises:
            ModelRegistrationError: If an error occurs during model registration.
                                    e.g. the directory cannot be created.
        """
        version = f"v_{int(time.time())}"
        version_dir = self.base_dir / version

        try:
            version_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            raise ModelRegistrationError(f"Version Directory already exists: {version_dir}") from e

        logger.info("Created version directory: %s", version_dir)
        return version, version_dir

    def save_pickle(self, obj: Any, version_dir: Path, filename: str) -> Path:
        """Serialize and save an object as a pickle file.

                Args:
                    obj: Python object to serialize.
                    version_dir: Target version directory.
                    filename: Output filename (e.g., 'model.pkl').

                Returns:
                    Path to the saved file.

                Raises:
                    ModelRegistrationError: If serialization fails.
                """
        path = version_dir / filename
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            logger.info("Saved artifact: %s", path)
        except Exception as e:
            raise ModelRegistrationError(f"Failed to save {filename}: {e}") from e
        return path

    def load_pickle(self, path: Path) -> Any:
        """Load a pickle artifact from disk.

        Args:
            path: Path to pickle artifact.

        Returns:
            Deserialized Python object.

        Raises:
            ModelRegistrationError: If serialization fails.
                                    If the file is missing or corrupted.
        """
        if not path.exists():
            raise ModelRegistrationError(f"Artifact does not exist: {path}")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logger.info("Loaded artifact: %s", path)
            return obj
        except Exception as e:
            raise ModelRegistrationError(f"Failed to load {path}: {e}") from e

    def save_json(self, data: Any, version_dir: Path, filename: str) -> Path:
        """Serialize and save data as a JSON file.

        Args:
            data: Python object to serialized in JSON format.
            version_dir: Target version directory.
            filename: Output filename (e.g., 'model.json').

        Returns:
            Path to the saved file.

        Raises:
            ModelRegistrationError: If serialization fails.
        """
        path = version_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved artifact: %s", path)
        except Exception as e:
            raise ModelRegistrationError(f"Failed to save {filename}: {e}") from e
        return path

    def load_json(self, path: Path) -> Any:
        """Load a JSON artifact from disk.

        Args:
            path: Path to JSON artifact.

        Returns:
            Deserialized Python object.

        Raises:
            ModelRegistrationError: If serialization fails e.g. file is missing or corrupted or parsing issues.
        """
        if not path.exists():
            raise ModelRegistrationError(f"Artifact does not exist: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            raise ModelRegistrationError(f"Failed to load {path}: {e}") from e

    def get_latest_version_dir(self) -> Optional[Path]:
        """Find the most recently created version directory.

        Returns:
            Path to the most recently created version directory, or None if no version directory exists.
        """
        version_dirs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("v_")],
            key=lambda d:d.name,
            reverse=True,
        )
        if not version_dirs:
            logger.warning("No version directories found in: %s", self.base_dir)
            return None

        logger.info("Latest version: %s", version_dirs[0].name)
        return version_dirs[0]

    def is_artifact_healthy(self,artifact_path: Path) -> bool:
        """Check whether an artifact file exists and is non-empty.

        Args:
            artifact_path: Path to artifact file.

        Returns:
            True if the artifact exists and is not empty, False otherwise.
        """
        return artifact_path.exists() and artifact_path.stat().st_size > 0






