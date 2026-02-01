"""Tests for trading strategy module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.strategy import (
    SignalType,
    Signal,
    Trade,
    VolatilityStrategy,
    SimpleVRPStrategy,
)


class TestSignalType:
    """Tests for SignalType enum."""

    def test_signal_values(self):
        """Test signal type values."""
        assert SignalType.SELL_VOL.value == -1
        assert SignalType.NEUTRAL.value == 0
        assert SignalType.BUY_VOL.value == 1

    def test_signal_names(self):
        """Test signal type names."""
        assert SignalType.SELL_VOL.name == "SELL_VOL"
        assert SignalType.NEUTRAL.name == "NEUTRAL"
        assert SignalType.BUY_VOL.name == "BUY_VOL"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a signal."""
        signal = Signal(
            date=date(2024, 1, 15),
            signal_type=SignalType.SELL_VOL,
            strength=0.8,
            iv=0.20,
            predicted_rv=0.15,
            vrp=0.05,
        )

        assert signal.date == date(2024, 1, 15)
        assert signal.signal_type == SignalType.SELL_VOL
        assert signal.strength == 0.8
        assert signal.vrp == 0.05

    def test_signal_with_metadata(self):
        """Test signal with metadata."""
        signal = Signal(
            date=date(2024, 1, 15),
            signal_type=SignalType.SELL_VOL,
            strength=0.8,
            iv=0.20,
            predicted_rv=0.15,
            vrp=0.05,
            metadata={"model": "xgboost", "confidence": 0.95},
        )

        assert signal.metadata["model"] == "xgboost"


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            entry_date=date(2024, 1, 15),
            exit_date=date(2024, 2, 5),
            signal_type=SignalType.SELL_VOL,
            entry_iv=0.20,
            exit_rv=0.15,
            pnl=0.25,
            holding_days=21,
        )

        assert trade.entry_date == date(2024, 1, 15)
        assert trade.exit_date == date(2024, 2, 5)
        assert trade.pnl == 0.25


class TestVolatilityStrategy:
    """Tests for VolatilityStrategy class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock volatility model."""
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([0.15] * 100))
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            },
            index=dates,
        )

        iv = pd.Series(np.random.uniform(0.18, 0.25, 100), index=dates)
        rv = pd.Series(np.random.uniform(0.12, 0.18, 100), index=dates)

        return X, iv, rv

    @pytest.fixture
    def strategy(self, mock_model):
        """Create a volatility strategy."""
        return VolatilityStrategy(
            model=mock_model,
            vrp_threshold=0.02,
            position_holding_days=21,
            use_signal_strength=True,
            max_positions=1,
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.vrp_threshold == 0.02
        assert strategy.position_holding_days == 21
        assert strategy.max_positions == 1

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation."""
        X, iv, rv = sample_data

        signals_df = strategy.generate_signals(X, iv)

        assert len(signals_df) == len(X)
        assert "signal" in signals_df.columns
        assert "signal_name" in signals_df.columns
        assert "strength" in signals_df.columns
        assert "vrp" in signals_df.columns

    def test_generate_signals_high_vrp(self, mock_model):
        """Test signal generation with high VRP."""
        # Model predicts low RV
        mock_model.predict = MagicMock(return_value=np.array([0.12] * 10))
        strategy = VolatilityStrategy(model=mock_model, vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(10)}, index=dates)
        iv = pd.Series([0.20] * 10, index=dates)  # High IV

        signals_df = strategy.generate_signals(X, iv)

        # Should have SELL_VOL signals (IV > predicted RV)
        sell_signals = signals_df[signals_df["signal"] == -1]
        assert len(sell_signals) > 0

    def test_generate_signals_low_vrp(self, mock_model):
        """Test signal generation with low VRP."""
        # Model predicts high RV
        mock_model.predict = MagicMock(return_value=np.array([0.25] * 10))
        strategy = VolatilityStrategy(model=mock_model, vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(10)}, index=dates)
        iv = pd.Series([0.15] * 10, index=dates)  # Low IV

        signals_df = strategy.generate_signals(X, iv)

        # Should have BUY_VOL signals (IV < predicted RV)
        buy_signals = signals_df[signals_df["signal"] == 1]
        assert len(buy_signals) > 0

    def test_generate_signals_neutral(self, mock_model):
        """Test signal generation with neutral VRP."""
        # Model predicts same as IV
        mock_model.predict = MagicMock(return_value=np.array([0.20] * 10))
        strategy = VolatilityStrategy(model=mock_model, vrp_threshold=0.05)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(10)}, index=dates)
        iv = pd.Series([0.20] * 10, index=dates)  # Same as predicted

        signals_df = strategy.generate_signals(X, iv)

        # Should be all NEUTRAL (VRP = 0)
        neutral_signals = signals_df[signals_df["signal"] == 0]
        assert len(neutral_signals) == 10

    def test_backtest(self, strategy, sample_data):
        """Test backtesting."""
        X, iv, rv = sample_data

        signals_df = strategy.generate_signals(X, iv)
        trades_df = strategy.backtest(signals_df, rv)

        if not trades_df.empty:
            assert "entry_date" in trades_df.columns
            assert "exit_date" in trades_df.columns
            assert "pnl" in trades_df.columns
            assert "cumulative_pnl" in trades_df.columns

    def test_backtest_no_signals(self, mock_model):
        """Test backtest with no trades."""
        mock_model.predict = MagicMock(return_value=np.array([0.20] * 30))
        strategy = VolatilityStrategy(model=mock_model, vrp_threshold=0.10)

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(30)}, index=dates)
        iv = pd.Series([0.20] * 30, index=dates)
        rv = pd.Series([0.20] * 30, index=dates)

        signals_df = strategy.generate_signals(X, iv)
        trades_df = strategy.backtest(signals_df, rv)

        # All signals should be neutral
        assert trades_df.empty or len(trades_df) == 0

    def test_backtest_sell_vol_profit(self, mock_model):
        """Test SELL_VOL trade with profit."""
        mock_model.predict = MagicMock(return_value=np.array([0.12] * 50))
        strategy = VolatilityStrategy(
            model=mock_model,
            vrp_threshold=0.02,
            position_holding_days=21,
            use_signal_strength=False,
        )

        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(50)}, index=dates)
        iv = pd.Series([0.20] * 50, index=dates)
        rv = pd.Series([0.15] * 50, index=dates)  # RV < IV = profit

        signals_df = strategy.generate_signals(X, iv)
        trades_df = strategy.backtest(signals_df, rv)

        if not trades_df.empty:
            # SELL_VOL profit when IV > RV
            assert trades_df["pnl"].sum() > 0

    def test_get_equity_curve(self, strategy, sample_data):
        """Test equity curve generation."""
        X, iv, rv = sample_data

        signals_df = strategy.generate_signals(X, iv)
        equity_df = strategy.get_equity_curve(signals_df, rv)

        if not equity_df.empty:
            assert "equity" in equity_df.columns
            assert "returns" in equity_df.columns
            assert equity_df["equity"].iloc[0] == 100000.0

    def test_signals_stored(self, strategy, sample_data):
        """Test that signals are stored in strategy."""
        X, iv, rv = sample_data

        strategy.generate_signals(X, iv)

        assert len(strategy.signals) == len(X)
        assert all(isinstance(s, Signal) for s in strategy.signals)

    def test_trades_stored(self, strategy, sample_data):
        """Test that trades are stored in strategy."""
        X, iv, rv = sample_data

        signals_df = strategy.generate_signals(X, iv)
        strategy.backtest(signals_df, rv)

        # trades list is populated
        assert isinstance(strategy.trades, list)


class TestSimpleVRPStrategy:
    """Tests for SimpleVRPStrategy class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        iv = pd.Series(np.random.uniform(0.18, 0.25, 100), index=dates)
        rv = pd.Series(np.random.uniform(0.12, 0.18, 100), index=dates)

        return iv, rv

    @pytest.fixture
    def strategy(self):
        """Create a simple VRP strategy."""
        return SimpleVRPStrategy(
            vrp_threshold=0.02,
            position_holding_days=21,
            lookback_days=21,
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.vrp_threshold == 0.02
        assert strategy.position_holding_days == 21
        assert strategy.lookback_days == 21

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation."""
        iv, rv = sample_data

        signals_df = strategy.generate_signals(iv, rv)

        assert len(signals_df) == len(iv)
        assert "signal" in signals_df.columns
        assert "vrp" in signals_df.columns

    def test_generate_signals_high_vrp(self):
        """Test signal generation with high VRP."""
        strategy = SimpleVRPStrategy(vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        iv = pd.Series([0.25] * 10, index=dates)
        rv = pd.Series([0.15] * 10, index=dates)  # VRP = 0.10

        signals_df = strategy.generate_signals(iv, rv)

        # All should be SELL_VOL
        assert all(signals_df["signal"] == -1)

    def test_generate_signals_negative_vrp(self):
        """Test signal generation with negative VRP."""
        strategy = SimpleVRPStrategy(vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        iv = pd.Series([0.15] * 10, index=dates)
        rv = pd.Series([0.25] * 10, index=dates)  # VRP = -0.10

        signals_df = strategy.generate_signals(iv, rv)

        # All should be BUY_VOL
        assert all(signals_df["signal"] == 1)

    def test_generate_signals_with_nan(self):
        """Test signal generation handles NaN."""
        strategy = SimpleVRPStrategy(vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        iv = pd.Series([0.20] * 10, index=dates)
        rv = pd.Series([0.15] * 10, index=dates)
        iv.iloc[0] = np.nan  # Add NaN

        signals_df = strategy.generate_signals(iv, rv)

        # First signal should be NEUTRAL due to NaN
        assert signals_df["signal"].iloc[0] == 0

    def test_signal_strength_calculation(self):
        """Test that signal strength is calculated correctly."""
        strategy = SimpleVRPStrategy(vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        iv = pd.Series([0.25, 0.20, 0.22], index=dates)
        rv = pd.Series([0.15, 0.18, 0.19], index=dates)

        signals_df = strategy.generate_signals(iv, rv)

        # Strength should be between 0 and 1
        assert all(signals_df["strength"] >= 0)
        assert all(signals_df["strength"] <= 1)

        # Higher VRP should have higher strength
        assert signals_df["strength"].iloc[0] > signals_df["strength"].iloc[1]

    def test_signals_stored(self, strategy, sample_data):
        """Test that signals are stored."""
        iv, rv = sample_data

        strategy.generate_signals(iv, rv)

        assert len(strategy.signals) == len(iv)


class TestStrategyEdgeCases:
    """Edge case tests for strategies."""

    def test_empty_data(self):
        """Test handling empty data."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([]))

        strategy = VolatilityStrategy(model=mock_model)

        dates = pd.DatetimeIndex([])
        X = pd.DataFrame({"feature1": []}, index=dates)
        iv = pd.Series([], index=dates, dtype=float)

        signals_df = strategy.generate_signals(X, iv)

        assert len(signals_df) == 0

    def test_single_data_point(self):
        """Test handling single data point."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.15]))

        strategy = VolatilityStrategy(model=mock_model, vrp_threshold=0.02)

        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        X = pd.DataFrame({"feature1": [0]}, index=dates)
        iv = pd.Series([0.20], index=dates)

        signals_df = strategy.generate_signals(X, iv)

        assert len(signals_df) == 1

    def test_backtest_insufficient_data(self):
        """Test backtest with insufficient data for holding period."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.12] * 5))

        strategy = VolatilityStrategy(
            model=mock_model,
            vrp_threshold=0.02,
            position_holding_days=21,  # Longer than data
        )

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        X = pd.DataFrame({"feature1": np.zeros(5)}, index=dates)
        iv = pd.Series([0.20] * 5, index=dates)
        rv = pd.Series([0.15] * 5, index=dates)

        signals_df = strategy.generate_signals(X, iv)
        trades_df = strategy.backtest(signals_df, rv)

        # Should have no trades due to insufficient holding period
        assert trades_df.empty
