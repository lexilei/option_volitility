"""Parquet storage utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


class ParquetStorage:
    """Handler for reading and writing Parquet files."""

    def __init__(self, base_dir: str | Path = "data"):
        """Initialize the storage handler.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.models_dir = self.base_dir / "models"

        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str, directory: str = "raw") -> Path:
        """Get the full path for a storage key.

        Args:
            key: Storage key (can include subdirectories)
            directory: Base directory (raw, processed, models)

        Returns:
            Full path to the parquet file
        """
        if directory == "raw":
            base = self.raw_dir
        elif directory == "processed":
            base = self.processed_dir
        elif directory == "models":
            base = self.models_dir
        else:
            base = self.base_dir / directory

        # Ensure key ends with .parquet
        if not key.endswith(".parquet"):
            key = f"{key}.parquet"

        return base / key

    def save(
        self,
        df: pd.DataFrame,
        key: str,
        directory: str = "raw",
        compression: str = "snappy",
        **kwargs: Any,
    ) -> Path:
        """Save a DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            key: Storage key
            directory: Target directory
            compression: Compression algorithm
            **kwargs: Additional arguments for pyarrow

        Returns:
            Path to the saved file
        """
        path = self._get_path(key, directory)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)

        # Write to parquet
        pq.write_table(table, path, compression=compression, **kwargs)

        logger.debug(f"Saved {len(df)} rows to {path}")
        return path

    def load(
        self,
        key: str,
        directory: str = "raw",
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> pd.DataFrame | None:
        """Load a DataFrame from Parquet format.

        Args:
            key: Storage key
            directory: Source directory
            columns: Columns to load (None for all)
            filters: Row group filters

        Returns:
            DataFrame or None if file doesn't exist
        """
        path = self._get_path(key, directory)

        if not path.exists():
            logger.debug(f"File not found: {path}")
            return None

        try:
            df = pd.read_parquet(path, columns=columns, filters=filters)
            logger.debug(f"Loaded {len(df)} rows from {path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def exists(self, key: str, directory: str = "raw") -> bool:
        """Check if a file exists.

        Args:
            key: Storage key
            directory: Directory to check

        Returns:
            True if file exists
        """
        path = self._get_path(key, directory)
        return path.exists()

    def delete(self, key: str, directory: str = "raw") -> bool:
        """Delete a file.

        Args:
            key: Storage key
            directory: Directory

        Returns:
            True if file was deleted
        """
        path = self._get_path(key, directory)

        if path.exists():
            path.unlink()
            logger.debug(f"Deleted {path}")
            return True

        return False

    def list_files(self, directory: str = "raw", pattern: str = "**/*.parquet") -> list[Path]:
        """List all parquet files in a directory.

        Args:
            directory: Directory to list
            pattern: Glob pattern

        Returns:
            List of file paths
        """
        if directory == "raw":
            base = self.raw_dir
        elif directory == "processed":
            base = self.processed_dir
        elif directory == "models":
            base = self.models_dir
        else:
            base = self.base_dir / directory

        return list(base.glob(pattern))

    def get_metadata(self, key: str, directory: str = "raw") -> dict[str, Any] | None:
        """Get metadata from a parquet file.

        Args:
            key: Storage key
            directory: Directory

        Returns:
            Dictionary with file metadata
        """
        path = self._get_path(key, directory)

        if not path.exists():
            return None

        try:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata

            return {
                "num_rows": metadata.num_rows,
                "num_columns": metadata.num_columns,
                "num_row_groups": metadata.num_row_groups,
                "created_by": metadata.created_by,
                "format_version": metadata.format_version,
                "serialized_size": metadata.serialized_size,
                "schema": parquet_file.schema_arrow.to_string(),
            }
        except Exception as e:
            logger.error(f"Error reading metadata from {path}: {e}")
            return None

    def append(
        self,
        df: pd.DataFrame,
        key: str,
        directory: str = "raw",
        dedupe_column: str | None = None,
    ) -> Path:
        """Append data to an existing parquet file.

        Args:
            df: DataFrame to append
            key: Storage key
            directory: Target directory
            dedupe_column: Column to use for deduplication

        Returns:
            Path to the saved file
        """
        existing = self.load(key, directory)

        if existing is not None:
            df = pd.concat([existing, df], ignore_index=True)

            if dedupe_column and dedupe_column in df.columns:
                df = df.drop_duplicates(subset=[dedupe_column], keep="last")

        return self.save(df, key, directory)

    def save_processed(
        self,
        df: pd.DataFrame,
        key: str,
        compression: str = "snappy",
    ) -> Path:
        """Convenience method to save to processed directory.

        Args:
            df: DataFrame to save
            key: Storage key
            compression: Compression algorithm

        Returns:
            Path to the saved file
        """
        return self.save(df, key, directory="processed", compression=compression)

    def load_processed(
        self,
        key: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Convenience method to load from processed directory.

        Args:
            key: Storage key
            columns: Columns to load

        Returns:
            DataFrame or None
        """
        return self.load(key, directory="processed", columns=columns)
