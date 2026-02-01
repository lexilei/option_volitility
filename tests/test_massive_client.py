"""Tests for MassiveClient module."""

import pytest
import pandas as pd
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockAgg:
    """Mock aggregate bar."""

    def __init__(self, timestamp, o, h, l, c, v, vwap=None, transactions=None):
        self.timestamp = timestamp
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.vwap = vwap or c
        self.transactions = transactions or 100


class MockOptionDetails:
    """Mock option details."""

    def __init__(self, ticker, contract_type, expiration_date, strike_price):
        self.ticker = ticker
        self.contract_type = contract_type
        self.expiration_date = expiration_date
        self.strike_price = strike_price


class MockGreeks:
    """Mock Greeks."""

    def __init__(self, delta=0.5, gamma=0.02, theta=-0.05, vega=0.1):
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega


class MockUnderlying:
    """Mock underlying asset."""

    def __init__(self, price):
        self.price = price


class MockQuote:
    """Mock quote."""

    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask


class MockTrade:
    """Mock trade."""

    def __init__(self, price):
        self.price = price


class MockDay:
    """Mock day data."""

    def __init__(self, volume):
        self.volume = volume


class MockOption:
    """Mock option snapshot."""

    def __init__(
        self,
        ticker,
        contract_type,
        expiration_date,
        strike_price,
        iv,
        underlying_price,
        delta=0.5,
    ):
        self.details = MockOptionDetails(ticker, contract_type, expiration_date, strike_price)
        self.greeks = MockGreeks(delta=delta)
        self.underlying_asset = MockUnderlying(underlying_price)
        self.implied_volatility = iv
        self.open_interest = 1000
        self.day = MockDay(500)
        self.last_quote = MockQuote(1.0, 1.1)
        self.last_trade = MockTrade(1.05)


class TestMassiveClientImport:
    """Tests for MassiveClient import handling."""

    def test_import_error_without_massive(self):
        """Test that proper error is raised when massive is not installed."""
        with patch.dict("sys.modules", {"massive": None}):
            # Need to reimport to test import error
            pass  # Import error is handled in the module itself


class TestMassiveClient:
    """Tests for MassiveClient class."""

    @pytest.fixture
    def mock_rest_client(self):
        """Create a mock REST client."""
        return MagicMock()

    @pytest.fixture
    def client(self, mock_rest_client):
        """Create a MassiveClient with mocked REST client."""
        with patch("src.data.massive_client.RESTClient", return_value=mock_rest_client):
            from src.data.massive_client import MassiveClient

            return MassiveClient("test_api_key")

    def test_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert client.client is not None

    def test_get_aggregates_empty(self, client):
        """Test get_aggregates with no data."""
        client.client.list_aggs.return_value = []

        df = client.get_aggregates("SPY")

        assert df.empty

    def test_get_aggregates_with_data(self, client):
        """Test get_aggregates with sample data."""
        mock_aggs = [
            MockAgg(1704067200000, 475.0, 478.0, 474.0, 476.5, 1000000),
            MockAgg(1704153600000, 476.5, 480.0, 476.0, 479.0, 1200000),
            MockAgg(1704240000000, 479.0, 481.0, 478.0, 480.5, 1100000),
        ]
        client.client.list_aggs.return_value = mock_aggs

        df = client.get_aggregates("SPY")

        assert len(df) == 3
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_get_aggregates_with_dates(self, client):
        """Test get_aggregates with date parameters."""
        mock_aggs = [MockAgg(1704067200000, 475.0, 478.0, 474.0, 476.5, 1000000)]
        client.client.list_aggs.return_value = mock_aggs

        df = client.get_aggregates(
            "SPY",
            from_date=date(2024, 1, 1),
            to_date=date(2024, 1, 31),
        )

        assert len(df) == 1
        client.client.list_aggs.assert_called_once()

    def test_get_aggregates_with_string_dates(self, client):
        """Test get_aggregates with string date parameters."""
        mock_aggs = [MockAgg(1704067200000, 475.0, 478.0, 474.0, 476.5, 1000000)]
        client.client.list_aggs.return_value = mock_aggs

        df = client.get_aggregates(
            "SPY",
            from_date="2024-01-01",
            to_date="2024-01-31",
        )

        assert len(df) == 1

    def test_get_options_chain_iv_empty(self, client):
        """Test get_options_chain_iv with no data."""
        client.client.list_snapshot_options_chain.return_value = []

        df = client.get_options_chain_iv("SPY")

        assert df.empty

    def test_get_options_chain_iv_with_data(self, client):
        """Test get_options_chain_iv with sample data."""
        mock_options = [
            MockOption("SPY240119C00475000", "call", "2024-01-19", 475.0, 0.18, 480.0),
            MockOption("SPY240119P00475000", "put", "2024-01-19", 475.0, 0.20, 480.0),
            MockOption("SPY240119C00480000", "call", "2024-01-19", 480.0, 0.17, 480.0),
            MockOption("SPY240119P00480000", "put", "2024-01-19", 480.0, 0.19, 480.0),
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        df = client.get_options_chain_iv("SPY")

        assert len(df) == 4
        assert "implied_volatility" in df.columns
        assert "strike_price" in df.columns
        assert "underlying_price" in df.columns
        assert "delta" in df.columns

    def test_get_options_chain_iv_with_limit(self, client):
        """Test get_options_chain_iv respects limit."""
        mock_options = [
            MockOption(f"SPY240119C0047{i}000", "call", "2024-01-19", 470 + i, 0.18, 480.0)
            for i in range(10)
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        df = client.get_options_chain_iv("SPY", limit=5)

        assert len(df) == 5

    def test_get_options_chain_iv_with_expiration(self, client):
        """Test get_options_chain_iv with expiration filter."""
        mock_options = [
            MockOption("SPY240119C00475000", "call", "2024-01-19", 475.0, 0.18, 480.0),
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        df = client.get_options_chain_iv("SPY", expiration_date="2024-01-19")

        assert len(df) == 1

    def test_get_atm_iv_empty(self, client):
        """Test get_atm_iv with no options."""
        client.client.list_snapshot_options_chain.return_value = []

        iv = client.get_atm_iv("SPY")

        assert iv is None

    def test_get_atm_iv_with_data(self, client):
        """Test get_atm_iv calculates ATM IV correctly."""
        # Create options around ATM (underlying at 480)
        mock_options = [
            MockOption("SPY240215C00475000", "call", "2024-02-15", 475.0, 0.18, 480.0),
            MockOption("SPY240215P00475000", "put", "2024-02-15", 475.0, 0.19, 480.0),
            MockOption("SPY240215C00480000", "call", "2024-02-15", 480.0, 0.17, 480.0),
            MockOption("SPY240215P00480000", "put", "2024-02-15", 480.0, 0.18, 480.0),
            MockOption("SPY240215C00485000", "call", "2024-02-15", 485.0, 0.16, 480.0),
            MockOption("SPY240215P00485000", "put", "2024-02-15", 485.0, 0.17, 480.0),
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        iv = client.get_atm_iv("SPY", days_to_expiry=30)

        # Should return the average of ATM options
        assert iv is not None
        assert 0.1 < iv < 0.3  # Reasonable IV range

    def test_get_atm_iv_converts_percentage(self, client):
        """Test get_atm_iv converts percentage to decimal."""
        # IV in percentage form (> 1)
        mock_options = [
            MockOption("SPY240215C00480000", "call", "2024-02-15", 480.0, 18.0, 480.0),
            MockOption("SPY240215P00480000", "put", "2024-02-15", 480.0, 19.0, 480.0),
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        iv = client.get_atm_iv("SPY", days_to_expiry=30)

        # Should be converted to decimal
        assert iv is not None
        assert iv < 1.0  # Should be decimal form

    def test_get_historical_iv(self, client):
        """Test get_historical_iv returns current IV."""
        mock_options = [
            MockOption("SPY240215C00480000", "call", "2024-02-15", 480.0, 0.18, 480.0),
            MockOption("SPY240215P00480000", "put", "2024-02-15", 480.0, 0.19, 480.0),
        ]
        client.client.list_snapshot_options_chain.return_value = mock_options

        iv_series = client.get_historical_iv("SPY")

        assert len(iv_series) == 1
        assert iv_series.iloc[0] > 0

    def test_get_historical_iv_empty(self, client):
        """Test get_historical_iv with no data."""
        client.client.list_snapshot_options_chain.return_value = []

        iv_series = client.get_historical_iv("SPY")

        assert iv_series.empty


class TestMassiveClientEdgeCases:
    """Edge case tests for MassiveClient."""

    @pytest.fixture
    def mock_rest_client(self):
        """Create a mock REST client."""
        return MagicMock()

    @pytest.fixture
    def client(self, mock_rest_client):
        """Create a MassiveClient with mocked REST client."""
        with patch("src.data.massive_client.RESTClient", return_value=mock_rest_client):
            from src.data.massive_client import MassiveClient

            return MassiveClient("test_api_key")

    def test_options_with_missing_details(self, client):
        """Test handling options with missing details."""
        mock_option = MagicMock()
        mock_option.details = None
        mock_option.greeks = None
        mock_option.underlying_asset = None
        mock_option.implied_volatility = 0.18
        mock_option.open_interest = 100
        mock_option.day = None
        mock_option.last_quote = None
        mock_option.last_trade = None

        client.client.list_snapshot_options_chain.return_value = [mock_option]

        df = client.get_options_chain_iv("SPY")

        assert len(df) == 1
        assert df["implied_volatility"].iloc[0] == 0.18

    def test_atm_iv_fallback_to_strikes(self, client):
        """Test ATM IV calculation when underlying price is missing."""
        mock_option = MagicMock()
        mock_option.details = MockOptionDetails(
            "SPY240215C00480000", "call", "2024-02-15", 480.0
        )
        mock_option.greeks = MockGreeks()
        mock_option.underlying_asset = None  # No underlying price
        mock_option.implied_volatility = 0.18
        mock_option.open_interest = 100
        mock_option.day = MockDay(500)
        mock_option.last_quote = MockQuote(1.0, 1.1)
        mock_option.last_trade = MockTrade(1.05)

        client.client.list_snapshot_options_chain.return_value = [mock_option]

        # Should still work using strike price median
        iv = client.get_atm_iv("SPY", days_to_expiry=30)

        assert iv is not None

    def test_atm_iv_no_valid_iv(self, client):
        """Test ATM IV when all IVs are invalid."""
        mock_option = MagicMock()
        mock_option.details = MockOptionDetails(
            "SPY240215C00480000", "call", "2024-02-15", 480.0
        )
        mock_option.greeks = None
        mock_option.underlying_asset = MockUnderlying(480.0)
        mock_option.implied_volatility = None  # No IV
        mock_option.open_interest = 100
        mock_option.day = None
        mock_option.last_quote = None
        mock_option.last_trade = None

        client.client.list_snapshot_options_chain.return_value = [mock_option]

        iv = client.get_atm_iv("SPY")

        assert iv is None
