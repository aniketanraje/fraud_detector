"""Huggingface dataset ingestion with retry logic and SHA-256 integrity caching."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tornado.web import url
from urllib3.util.retry import Retry

from src.constants import DATA_RAW_DIR, HF_DATASET_URL
from src.domain.exceptions import DataIngestionError

logger: logging.Logger = logging.getLogger(__name__)

_HASH_SUFFIX: str = ".sha256"
_TIMEOUT_SECONDS: int = 60
_RETRY_ATTEMPTS: int = 3
_BACKOFF_FACTOR: float = 1.5


class DataIngestor:
    """Fetches and caches the credit card dataset from Huggingface.

    Implements retry-backed HTTP download with SHA-256 integrity checking.
    On subsequent runs, returns the cached file if the hash matches.

    Attributes:
        url: Source URL for the CSV dataset.
        cache_dir: Local directory for caching downloaded files.
        filename: Name to save the downloaded CSV file.
    """
    def __init__(
            self,
            url: str = HF_DATASET_URL,
            cache_dir: Path = DATA_RAW_DIR,
            filename: str = "creditcard.csv",
    ) -> None:
        """Initializes DataIngestor.

        Args:
            url: Source URL for the CSV dataset.
            cache_dir: Local directory for caching downloaded files.
            filename: Name to save the downloaded CSV file.
        """
        self.url: str = url
        self.cache_dir: Path = Path(cache_dir)
        self.filename: str = filename
        self._cache_path: Path = self.cache_dir / filename
        self._hash_path: Path = self.cache_dir / f"{filename}{_HASH_SUFFIX}"
        self._session: requests.Session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Build a request session with exponential backoff retry strategy.

        Returns:
            A configured requests.Session object.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=_RETRY_ATTEMPTS,
            backoff_factor=_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _compute_sha256(self, path: Path) -> str:
        """Compute the SHA-256 hash of the CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


    def _is_cache_valid(self) -> bool:
        """Check whether a valid cached file exists with a matching hash.

        Returns:
            True if the cache is present and Hash matches, False otherwise.
        """
        if not self._cache_path.exists() or not self._hash_path.exists():
            return False
        stored_hash = self._hash_path.read_text(encoding="utf-8").strip()
        actual_hash = self._compute_sha256(self._cache_path)

        if stored_hash == actual_hash:
            logger.info("Cache hit — SHA-256 verified: %s", actual_hash[:16])
            return True

        logger.warning("Cache hash mismatch – re-downloading")
        return False

    def _download(self) -> None:
        """Download the dataset from the Huggingface and save to cache.

        Raises:
            DataIngestionError: If an error occurs while downloading the dataset or response is not 200.
        """
        logger.info("Downloading dataset from: %s", self.url)
        start: float = time.time()

        try:
            response = self._session.get(self.url, timeout=_TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise DataIngestionError(f"Download failed after {_RETRY_ATTEMPTS} times. {e}") from e

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with open(self._cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        digest = self._compute_sha256(self._cache_path)
        self._hash_path.write_text(digest, encoding="utf-8")

        elapsed = round(time.time() - start, 2)
        logger.info("Download complete — %ss | SHA-256: %s", elapsed, digest[:16])

    def fetch(self) -> pd.DataFrame:
        """Fetch the credit card dataset from local cache, if valid.

        Returns:
            A Pandas DataFrame object containing the credit card dataset.

        Raises:
            DataIngestionError: If an error occurs while downloading or the CSV cannot be parsed.
        """
        if not self._is_cache_valid():
            self._download()

        logger.info("Loading CSV from cache: %s", self._cache_path)

        try:
            df: pd.DataFrame = pd.read_csv(self._cache_path)
        except Exception as e:
            raise DataIngestionError(f"Failed to load CSV from cache: {e}") from e

        if df.empty:
            raise DataIngestionError("Loaded DataFrame is empty")

        logger.info("Dataset loaded — shape: %s", df.shape)
        return df

