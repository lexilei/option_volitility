"""Tests for paper trading module."""

import pytest
import json
import tempfile
from datetime import date
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.paper_trader import PaperTrader, Position, TradeRecord


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            position_id=1,
            entry_date="2024-01-15",
            position_type="SELL_VOL",
            entry_iv=0.20,
            entry_rv=0.15,
            entry_vrp=0.05,
            target_exit_date="2024-02-05",
            underlying_price=500.0,
        )

        assert pos.position_id == 1
        assert pos.position_type == "SELL_VOL"
        assert pos.entry_iv == 0.20
        assert pos.status == "OPEN"
        assert pos.pnl is None

    def test_position_to_dict(self):
        """Test converting position to dict."""
        pos = Position(
            position_id=1,
            entry_date="2024-01-15",
            position_type="SELL_VOL",
            entry_iv=0.20,
            entry_rv=0.15,
            entry_vrp=0.05,
            target_exit_date="2024-02-05",
            underlying_price=500.0,
        )

        d = pos.to_dict()
        assert isinstance(d, dict)
        assert d["position_id"] == 1
        assert d["position_type"] == "SELL_VOL"

    def test_position_from_dict(self):
        """Test creating position from dict."""
        data = {
            "position_id": 1,
            "entry_date": "2024-01-15",
            "position_type": "SELL_VOL",
            "entry_iv": 0.20,
            "entry_rv": 0.15,
            "entry_vrp": 0.05,
            "target_exit_date": "2024-02-05",
            "underlying_price": 500.0,
            "notional_value": 10000.0,
            "status": "OPEN",
            "exit_date": None,
            "exit_iv": None,
            "exit_rv": None,
            "pnl": None,
            "pnl_pct": None,
        }

        pos = Position.from_dict(data)
        assert pos.position_id == 1
        assert pos.entry_iv == 0.20


class TestPaperTrader:
    """Tests for PaperTrader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def trader(self, temp_dir):
        """Create a paper trader instance."""
        return PaperTrader(
            data_dir=temp_dir,
            initial_capital=100000.0,
            position_size=10000.0,
            max_positions=3,
            holding_days=21,
            vrp_threshold=0.02,
        )

    def test_initialization(self, trader):
        """Test trader initialization."""
        assert trader.initial_capital == 100000.0
        assert trader.cash == 100000.0
        assert trader.position_size == 10000.0
        assert trader.max_positions == 3
        assert len(trader.positions) == 0

    def test_get_open_positions_empty(self, trader):
        """Test getting open positions when none exist."""
        assert len(trader.get_open_positions()) == 0

    def test_should_open_position_high_vrp(self, trader):
        """Test signal generation with high VRP."""
        should_open, pos_type = trader.should_open_position(iv=0.25, rv=0.15)
        assert should_open is True
        assert pos_type == "SELL_VOL"

    def test_should_open_position_low_vrp(self, trader):
        """Test signal generation with low VRP."""
        should_open, pos_type = trader.should_open_position(iv=0.15, rv=0.25)
        assert should_open is True
        assert pos_type == "BUY_VOL"

    def test_should_open_position_neutral(self, trader):
        """Test signal generation with neutral VRP."""
        should_open, pos_type = trader.should_open_position(iv=0.20, rv=0.19)
        assert should_open is False
        assert pos_type == ""

    def test_open_position(self, trader):
        """Test opening a position."""
        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.25,
            rv=0.15,
            underlying_price=500.0,
        )

        assert pos is not None
        assert pos.position_id == 1
        assert pos.position_type == "SELL_VOL"
        assert pos.entry_iv == 0.25
        assert trader.cash == 90000.0  # 100000 - 10000
        assert len(trader.get_open_positions()) == 1

    def test_open_position_insufficient_cash(self, trader):
        """Test that position opening fails with insufficient cash."""
        trader.cash = 5000.0  # Less than position_size

        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.25,
            rv=0.15,
            underlying_price=500.0,
        )

        assert pos is None

    def test_max_positions_limit(self, trader):
        """Test that max positions are enforced."""
        # Open max positions
        for i in range(3):
            trader.open_position(
                position_type="SELL_VOL",
                current_date=date(2024, 1, 15),
                iv=0.25,
                rv=0.15,
                underlying_price=500.0,
            )

        assert len(trader.get_open_positions()) == 3

        # Try to open another - should fail due to max positions
        should_open, _ = trader.should_open_position(iv=0.25, rv=0.15)
        assert should_open is False

    def test_close_position_profit(self, trader):
        """Test closing a profitable position."""
        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.25,
            rv=0.15,
            underlying_price=500.0,
        )

        # IV decreased - profit for SELL_VOL
        trade = trader.close_position(
            pos,
            current_date=date(2024, 2, 5),
            iv=0.20,  # IV decreased by 5%
            rv=0.14,
        )

        assert trade.pnl > 0  # Profit
        assert pos.status == "CLOSED"
        assert trader.cash > 90000.0  # Got back more than invested

    def test_close_position_loss(self, trader):
        """Test closing a losing position."""
        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )

        # IV increased - loss for SELL_VOL
        trade = trader.close_position(
            pos,
            current_date=date(2024, 2, 5),
            iv=0.25,  # IV increased by 5%
            rv=0.16,
        )

        assert trade.pnl < 0  # Loss
        assert pos.status == "CLOSED"

    def test_get_equity(self, trader):
        """Test equity calculation."""
        # Initial equity
        equity = trader.get_equity(current_iv=0.20, current_rv=0.15)
        assert equity == 100000.0

        # Open a position
        trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )

        # Equity should still be ~100000 (cash + position value)
        equity = trader.get_equity(current_iv=0.20, current_rv=0.15)
        assert equity == 100000.0  # No change in IV = no P&L

    def test_get_equity_with_unrealized_pnl(self, trader):
        """Test equity with unrealized P&L."""
        trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )

        # IV decreased - unrealized profit
        equity = trader.get_equity(current_iv=0.15, current_rv=0.12)
        assert equity > 100000.0  # Should have unrealized profit

    def test_check_exits(self, trader):
        """Test automatic exit checking."""
        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )

        # Before exit date - no exits
        closed = trader.check_exits(
            current_date=date(2024, 1, 20),
            iv=0.18,
            rv=0.14,
        )
        assert len(closed) == 0
        assert pos.status == "OPEN"

        # After exit date - should exit
        closed = trader.check_exits(
            current_date=date(2024, 2, 10),  # After 21 days
            iv=0.18,
            rv=0.14,
        )
        assert len(closed) == 1
        assert pos.status == "CLOSED"

    def test_record_snapshot(self, trader):
        """Test recording daily snapshot."""
        snapshot = trader.record_snapshot(
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
        )

        assert snapshot.equity == 100000.0
        assert snapshot.open_positions == 0
        assert len(trader.snapshots) == 1

    def test_run_daily_update(self, trader):
        """Test full daily update cycle."""
        summary = trader.run_daily_update(
            current_date=date(2024, 1, 15),
            iv=0.25,  # High VRP should trigger trade
            rv=0.15,
            underlying_price=500.0,
        )

        assert summary["new_positions"] == 1
        assert summary["open_positions"] == 1
        assert summary["equity"] == 100000.0

    def test_save_and_load_state(self, temp_dir):
        """Test state persistence."""
        # Create trader and open position
        trader1 = PaperTrader(data_dir=temp_dir)
        trader1.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )
        trader1._save_state()

        # Create new trader - should load state
        trader2 = PaperTrader(data_dir=temp_dir)
        assert trader2.cash == 90000.0
        assert len(trader2.positions) == 1

    def test_reset(self, trader):
        """Test resetting trader state."""
        # Open position
        trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )
        trader.record_snapshot(date(2024, 1, 15), 0.20, 0.15)

        # Reset
        trader.reset()

        assert trader.cash == 100000.0
        assert len(trader.positions) == 0
        assert len(trader.trades) == 0
        assert len(trader.snapshots) == 0

    def test_get_performance_summary_empty(self, trader):
        """Test performance summary with no data."""
        summary = trader.get_performance_summary()
        assert summary == {}

    def test_get_performance_summary(self, trader):
        """Test performance summary with data."""
        # Record some snapshots
        trader.record_snapshot(date(2024, 1, 15), 0.20, 0.15)
        trader.record_snapshot(date(2024, 1, 16), 0.21, 0.15)

        summary = trader.get_performance_summary()

        assert "total_return" in summary
        assert "trading_days" in summary
        assert summary["trading_days"] == 2

    def test_get_equity_curve(self, trader):
        """Test getting equity curve."""
        trader.record_snapshot(date(2024, 1, 15), 0.20, 0.15)
        trader.record_snapshot(date(2024, 1, 16), 0.21, 0.15)

        df = trader.get_equity_curve()

        assert len(df) == 2
        assert "equity" in df.columns

    def test_get_trades_df_empty(self, trader):
        """Test getting trades DataFrame when empty."""
        df = trader.get_trades_df()
        assert df.empty

    def test_get_trades_df(self, trader):
        """Test getting trades DataFrame."""
        # Open and close a position
        pos = trader.open_position(
            position_type="SELL_VOL",
            current_date=date(2024, 1, 15),
            iv=0.20,
            rv=0.15,
            underlying_price=500.0,
        )
        trader.close_position(pos, date(2024, 2, 5), 0.18, 0.14)

        df = trader.get_trades_df()

        assert len(df) == 1
        assert "pnl" in df.columns
        assert "entry_date" in df.columns
