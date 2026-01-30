"""Tests for data modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.storage import ParquetStorage


class TestParquetStorage:
    """Tests for ParquetStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a storage instance."""
        return ParquetStorage(temp_dir)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date": dates,
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(105, 115, 100),
            "low": np.random.uniform(95, 105, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.randint(1000000, 5000000, 100),
        }).set_index("date")

    def test_save_and_load(self, storage, sample_df):
        """Test saving and loading a DataFrame."""
        storage.save(sample_df, "test_data")
        loaded = storage.load("test_data")

        assert loaded is not None
        assert len(loaded) == len(sample_df)
        assert list(loaded.columns) == list(sample_df.columns)

    def test_exists(self, storage, sample_df):
        """Test file existence check."""
        assert not storage.exists("nonexistent")

        storage.save(sample_df, "test_data")
        assert storage.exists("test_data")

    def test_delete(self, storage, sample_df):
        """Test file deletion."""
        storage.save(sample_df, "test_data")
        assert storage.exists("test_data")

        storage.delete("test_data")
        assert not storage.exists("test_data")

    def test_list_files(self, storage, sample_df):
        """Test listing files."""
        storage.save(sample_df, "data1")
        storage.save(sample_df, "data2")

        files = storage.list_files()
        assert len(files) == 2

    def test_load_with_columns(self, storage, sample_df):
        """Test loading specific columns."""
        storage.save(sample_df, "test_data")
        loaded = storage.load("test_data", columns=["open", "close"])

        assert loaded is not None
        assert list(loaded.columns) == ["open", "close"]

    def test_append(self, storage, sample_df):
        """Test appending data."""
        storage.save(sample_df, "test_data")

        # Create additional data
        new_dates = pd.date_range(start="2023-04-11", periods=50, freq="D")
        new_df = pd.DataFrame({
            "date": new_dates,
            "open": np.random.uniform(100, 110, 50),
            "high": np.random.uniform(105, 115, 50),
            "low": np.random.uniform(95, 105, 50),
            "close": np.random.uniform(100, 110, 50),
            "volume": np.random.randint(1000000, 5000000, 50),
        }).set_index("date")

        storage.append(new_df, "test_data")
        loaded = storage.load("test_data")

        assert loaded is not None
        assert len(loaded) == 150  # 100 + 50

    def test_get_metadata(self, storage, sample_df):
        """Test getting file metadata."""
        storage.save(sample_df, "test_data")
        metadata = storage.get_metadata("test_data")

        assert metadata is not None
        assert metadata["num_rows"] == 100
        assert metadata["num_columns"] == 5

    def test_load_nonexistent(self, storage):
        """Test loading nonexistent file returns None."""
        result = storage.load("nonexistent_file")
        assert result is None

    def test_nested_key(self, storage, sample_df):
        """Test saving with nested key path."""
        storage.save(sample_df, "prices/SPY/daily")
        assert storage.exists("prices/SPY/daily")

        loaded = storage.load("prices/SPY/daily")
        assert loaded is not None


class TestDataFetcher:
    """Tests for DataFetcher class."""

    def test_import(self):
        """Test that DataFetcher can be imported."""
        from src.data.data_fetcher import DataFetcher

        assert DataFetcher is not None


class TestPolygonClient:
    """Tests for PolygonClient class."""

    def test_import(self):
        """Test that PolygonClient can be imported."""
        from src.data.polygon_client import PolygonClient

        assert PolygonClient is not None

    def test_initialization(self):
        """Test client initialization."""
        from src.data.polygon_client import PolygonClient

        client = PolygonClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.rate_limit_delay == 0.25
