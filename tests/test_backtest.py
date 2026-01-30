"""Tests for backtest modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.metrics import (
    calculate_returns_metrics,
    calculate_drawdown_metrics,
    calculate_trade_metrics,
    calculate_all_metrics,
    PerformanceMetrics,
)
from src.backtest.position import PositionManager, Position, PositionType
from src.backtest.risk import RiskManager, RiskLimits


class TestPerformanceMetrics:
    """Tests for performance metrics calculations."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        n = 252
        returns = pd.Series(np.random.normal(0.0008, 0.012, n))
        return returns

    @pytest.fixture
    def sample_equity(self, sample_returns):
        """Generate sample equity curve."""
        equity = 100000 * (1 + sample_returns).cumprod()
        return equity

    @pytest.fixture
    def sample_trades(self):
        """Generate sample trades."""
        np.random.seed(42)
        n_trades = 20

        pnl = np.random.normal(500, 2000, n_trades)
        return pd.DataFrame({"pnl": pnl})

    def test_returns_metrics(self, sample_returns):
        """Test returns metrics calculation."""
        metrics = calculate_returns_metrics(sample_returns)

        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics

    def test_returns_metrics_empty(self):
        """Test with empty returns."""
        metrics = calculate_returns_metrics(pd.Series([], dtype=float))

        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0

    def test_drawdown_metrics(self, sample_equity):
        """Test drawdown metrics calculation."""
        metrics = calculate_drawdown_metrics(sample_equity)

        assert "max_drawdown" in metrics
        assert "avg_drawdown" in metrics
        assert "max_drawdown_duration" in metrics
        assert metrics["max_drawdown"] >= 0

    def test_trade_metrics(self, sample_trades):
        """Test trade metrics calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        assert metrics["total_trades"] == 20
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "avg_win" in metrics
        assert "avg_loss" in metrics

    def test_trade_metrics_empty(self):
        """Test with no trades."""
        metrics = calculate_trade_metrics(pd.DataFrame())

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0

    def test_calculate_all_metrics(self, sample_equity, sample_trades):
        """Test combined metrics calculation."""
        metrics = calculate_all_metrics(sample_equity, sample_trades)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 20

    def test_performance_metrics_to_dict(self, sample_equity, sample_trades):
        """Test converting metrics to dictionary."""
        metrics = calculate_all_metrics(sample_equity, sample_trades)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "sharpe_ratio" in metrics_dict


class TestPositionManager:
    """Tests for PositionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a position manager."""
        return PositionManager(
            initial_capital=100000,
            max_position_size=0.1,
            max_positions=3,
        )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.capital == 100000
        assert manager.max_positions == 3
        assert len(manager.open_positions) == 0

    def test_open_position(self, manager):
        """Test opening a position."""
        position = manager.open_position(
            entry_date=date(2024, 1, 15),
            position_type=PositionType.SHORT,
            entry_price=2.50,
            underlying_price=100.0,
            quantity=10,
        )

        assert position is not None
        assert position.position_id == 1
        assert position.is_open
        assert manager.num_open_positions == 1

    def test_close_position(self, manager):
        """Test closing a position."""
        position = manager.open_position(
            entry_date=date(2024, 1, 15),
            position_type=PositionType.SHORT,
            entry_price=2.50,
            underlying_price=100.0,
            quantity=10,
        )

        pnl = manager.close_position(
            position,
            exit_date=date(2024, 2, 15),
            exit_price=1.50,
        )

        # Short position: profit when price goes down
        assert pnl > 0
        assert not position.is_open
        assert manager.num_open_positions == 0

    def test_max_positions_limit(self, manager):
        """Test that max positions are enforced."""
        for i in range(3):
            manager.open_position(
                entry_date=date(2024, 1, 15),
                position_type=PositionType.SHORT,
                entry_price=2.50,
                underlying_price=100.0,
            )

        # Fourth position should fail
        position = manager.open_position(
            entry_date=date(2024, 1, 15),
            position_type=PositionType.SHORT,
            entry_price=2.50,
            underlying_price=100.0,
        )

        assert position is None
        assert manager.num_open_positions == 3

    def test_get_trades_df(self, manager):
        """Test getting trades as DataFrame."""
        # Open and close a position
        position = manager.open_position(
            entry_date=date(2024, 1, 15),
            position_type=PositionType.SHORT,
            entry_price=2.50,
            underlying_price=100.0,
            quantity=10,
        )
        manager.close_position(position, date(2024, 2, 15), 1.50)

        trades_df = manager.get_trades_df()

        assert len(trades_df) == 1
        assert "pnl" in trades_df.columns

    def test_get_summary(self, manager):
        """Test getting summary statistics."""
        # Open and close positions
        for i in range(3):
            position = manager.open_position(
                entry_date=date(2024, 1, 15),
                position_type=PositionType.SHORT,
                entry_price=2.50,
                underlying_price=100.0,
                quantity=10,
            )
            manager.close_position(
                position,
                date(2024, 2, 15),
                2.50 + (i - 1) * 0.5,  # Win, break-even, loss
            )

        summary = manager.get_summary()

        assert summary["total_trades"] == 3
        assert "win_rate" in summary
        assert "total_pnl" in summary


class TestRiskManager:
    """Tests for RiskManager class."""

    @pytest.fixture
    def position_manager(self):
        """Create a position manager."""
        return PositionManager(initial_capital=100000)

    @pytest.fixture
    def risk_manager(self, position_manager):
        """Create a risk manager."""
        limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.15,
            stop_loss_pct=0.5,
        )
        return RiskManager(position_manager, limits)

    def test_check_position_size(self, risk_manager):
        """Test position size check."""
        # Within limit
        assert risk_manager.check_position_size(5000)

        # Exceeds limit
        assert not risk_manager.check_position_size(15000)

    def test_can_open_position(self, risk_manager, position_manager):
        """Test comprehensive position opening check."""
        can_open, reason = risk_manager.can_open_position(5000)
        assert can_open
        assert reason == "OK"

    def test_alerts_logged(self, risk_manager):
        """Test that alerts are logged."""
        # Trigger an alert
        risk_manager.check_position_size(15000)

        alerts = risk_manager.get_alerts()
        assert len(alerts) > 0
