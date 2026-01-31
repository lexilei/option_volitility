"""Paper trading system for simulated trading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger


@dataclass
class Position:
    """Represents an open trading position."""

    position_id: int
    entry_date: str
    position_type: Literal["SELL_VOL", "BUY_VOL"]
    entry_iv: float
    entry_rv: float
    entry_vrp: float
    target_exit_date: str
    underlying_price: float
    notional_value: float = 10000.0  # Default position size
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    exit_date: str | None = None
    exit_iv: float | None = None
    exit_rv: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        return cls(**data)


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    trade_id: int
    entry_date: str
    exit_date: str
    position_type: Literal["SELL_VOL", "BUY_VOL"]
    entry_iv: float
    entry_rv: float
    exit_iv: float
    exit_rv: float
    pnl: float
    pnl_pct: float
    holding_days: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""

    date: str
    equity: float
    cash: float
    open_positions: int
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float
    cumulative_return: float

    def to_dict(self) -> dict:
        return asdict(self)


class PaperTrader:
    """Paper trading engine for simulated volatility trading."""

    def __init__(
        self,
        data_dir: str = "data",
        initial_capital: float = 100000.0,
        position_size: float = 10000.0,
        max_positions: int = 5,
        holding_days: int = 21,
        vrp_threshold: float = 0.02,
    ):
        """Initialize paper trader.

        Args:
            data_dir: Directory for storing trading data
            initial_capital: Starting capital
            position_size: Size per position
            max_positions: Maximum concurrent positions
            holding_days: Days to hold each position
            vrp_threshold: VRP threshold for signals
        """
        self.data_dir = Path(data_dir)
        self.trading_dir = self.data_dir / "paper_trading"
        self.trading_dir.mkdir(parents=True, exist_ok=True)

        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.holding_days = holding_days
        self.vrp_threshold = vrp_threshold

        # State
        self.cash = initial_capital
        self.positions: list[Position] = []
        self.trades: list[TradeRecord] = []
        self.snapshots: list[DailySnapshot] = []
        self.next_position_id = 1
        self.next_trade_id = 1

        # Load existing state
        self._load_state()

    def _state_file(self) -> Path:
        return self.trading_dir / "state.json"

    def _trades_file(self) -> Path:
        return self.trading_dir / "trades.json"

    def _snapshots_file(self) -> Path:
        return self.trading_dir / "snapshots.json"

    def _load_state(self) -> None:
        """Load existing state from files."""
        # Load state
        if self._state_file().exists():
            with open(self._state_file()) as f:
                state = json.load(f)
                self.cash = state.get("cash", self.initial_capital)
                self.next_position_id = state.get("next_position_id", 1)
                self.next_trade_id = state.get("next_trade_id", 1)
                self.positions = [
                    Position.from_dict(p) for p in state.get("positions", [])
                ]
            logger.info(f"Loaded state: cash={self.cash:.2f}, positions={len(self.positions)}")

        # Load trades
        if self._trades_file().exists():
            with open(self._trades_file()) as f:
                trades_data = json.load(f)
                self.trades = [TradeRecord(**t) for t in trades_data]
            logger.info(f"Loaded {len(self.trades)} historical trades")

        # Load snapshots
        if self._snapshots_file().exists():
            with open(self._snapshots_file()) as f:
                snapshots_data = json.load(f)
                self.snapshots = [DailySnapshot(**s) for s in snapshots_data]
            logger.info(f"Loaded {len(self.snapshots)} daily snapshots")

    def _save_state(self) -> None:
        """Save current state to files."""
        # Save state
        state = {
            "cash": self.cash,
            "next_position_id": self.next_position_id,
            "next_trade_id": self.next_trade_id,
            "positions": [p.to_dict() for p in self.positions],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file(), "w") as f:
            json.dump(state, f, indent=2)

        # Save trades
        with open(self._trades_file(), "w") as f:
            json.dump([t.to_dict() for t in self.trades], f, indent=2)

        # Save snapshots
        with open(self._snapshots_file(), "w") as f:
            json.dump([s.to_dict() for s in self.snapshots], f, indent=2)

        logger.info("Saved trading state")

    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == "OPEN"]

    def get_equity(self, current_iv: float, current_rv: float) -> float:
        """Calculate total equity including unrealized P&L."""
        # Equity = cash + notional value of open positions + unrealized P&L
        open_notional = sum(pos.notional_value for pos in self.get_open_positions())
        unrealized = self._calculate_unrealized_pnl(current_iv, current_rv)
        return self.cash + open_notional + unrealized

    def _calculate_unrealized_pnl(self, current_iv: float, current_rv: float) -> float:
        """Calculate unrealized P&L for open positions."""
        unrealized = 0.0
        for pos in self.get_open_positions():
            if pos.position_type == "SELL_VOL":
                # Profit when IV decreases or RV < IV
                pnl = (pos.entry_iv - current_iv) * pos.notional_value
            else:
                # Profit when IV increases
                pnl = (current_iv - pos.entry_iv) * pos.notional_value
            unrealized += pnl
        return unrealized

    def should_open_position(self, iv: float, rv: float) -> tuple[bool, str]:
        """Check if we should open a new position.

        Args:
            iv: Current implied volatility
            rv: Current realized volatility

        Returns:
            Tuple of (should_open, position_type)
        """
        vrp = iv - rv

        # Check position limits
        open_count = len(self.get_open_positions())
        if open_count >= self.max_positions:
            return False, ""

        # Check cash
        if self.cash < self.position_size:
            return False, ""

        # Generate signal
        if vrp > self.vrp_threshold:
            return True, "SELL_VOL"
        elif vrp < -self.vrp_threshold:
            return True, "BUY_VOL"

        return False, ""

    def open_position(
        self,
        position_type: Literal["SELL_VOL", "BUY_VOL"],
        current_date: date,
        iv: float,
        rv: float,
        underlying_price: float,
    ) -> Position | None:
        """Open a new position.

        Args:
            position_type: Type of position
            current_date: Entry date
            iv: Current implied volatility
            rv: Current realized volatility
            underlying_price: Current underlying price

        Returns:
            New position or None if cannot open
        """
        if self.cash < self.position_size:
            logger.warning("Insufficient cash to open position")
            return None

        # Calculate target exit date
        target_exit = pd.Timestamp(current_date) + pd.Timedelta(days=self.holding_days)
        # Adjust to next business day if weekend
        while target_exit.weekday() >= 5:
            target_exit += pd.Timedelta(days=1)

        position = Position(
            position_id=self.next_position_id,
            entry_date=str(current_date),
            position_type=position_type,
            entry_iv=iv,
            entry_rv=rv,
            entry_vrp=iv - rv,
            target_exit_date=str(target_exit.date()),
            underlying_price=underlying_price,
            notional_value=self.position_size,
        )

        self.positions.append(position)
        self.cash -= self.position_size
        self.next_position_id += 1

        logger.info(
            f"Opened {position_type} position #{position.position_id}: "
            f"IV={iv:.4f}, RV={rv:.4f}, VRP={iv-rv:.4f}"
        )

        return position

    def close_position(
        self,
        position: Position,
        current_date: date,
        iv: float,
        rv: float,
    ) -> TradeRecord:
        """Close an existing position.

        Args:
            position: Position to close
            current_date: Exit date
            iv: Current implied volatility
            rv: Current realized volatility

        Returns:
            Trade record
        """
        # Calculate P&L
        if position.position_type == "SELL_VOL":
            # Profit = (entry_IV - exit_IV) * notional
            # Simplified: profit when IV decreases
            pnl = (position.entry_iv - iv) * position.notional_value
        else:
            # Buy vol: profit when IV increases
            pnl = (iv - position.entry_iv) * position.notional_value

        pnl_pct = pnl / position.notional_value

        # Update position
        position.status = "CLOSED"
        position.exit_date = str(current_date)
        position.exit_iv = iv
        position.exit_rv = rv
        position.pnl = pnl
        position.pnl_pct = pnl_pct

        # Return cash + P&L
        self.cash += position.notional_value + pnl

        # Create trade record
        entry_date = datetime.strptime(position.entry_date, "%Y-%m-%d").date()
        holding_days = (current_date - entry_date).days

        trade = TradeRecord(
            trade_id=self.next_trade_id,
            entry_date=position.entry_date,
            exit_date=str(current_date),
            position_type=position.position_type,
            entry_iv=position.entry_iv,
            entry_rv=position.entry_rv,
            exit_iv=iv,
            exit_rv=rv,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
        )

        self.trades.append(trade)
        self.next_trade_id += 1

        logger.info(
            f"Closed position #{position.position_id}: "
            f"P&L=${pnl:.2f} ({pnl_pct:.2%})"
        )

        return trade

    def check_exits(self, current_date: date, iv: float, rv: float) -> list[TradeRecord]:
        """Check and close positions that have reached exit date.

        Args:
            current_date: Current date
            iv: Current implied volatility
            rv: Current realized volatility

        Returns:
            List of closed trades
        """
        closed_trades = []

        for position in self.get_open_positions():
            exit_date = datetime.strptime(position.target_exit_date, "%Y-%m-%d").date()
            if current_date >= exit_date:
                trade = self.close_position(position, current_date, iv, rv)
                closed_trades.append(trade)

        return closed_trades

    def record_snapshot(self, current_date: date, iv: float, rv: float) -> DailySnapshot:
        """Record daily portfolio snapshot.

        Args:
            current_date: Current date
            iv: Current implied volatility
            rv: Current realized volatility

        Returns:
            Daily snapshot
        """
        equity = self.get_equity(iv, rv)
        unrealized = self._calculate_unrealized_pnl(iv, rv)
        realized = sum(t.pnl for t in self.trades)

        # Calculate returns
        if self.snapshots:
            prev_equity = self.snapshots[-1].equity
            daily_return = (equity - prev_equity) / prev_equity
        else:
            daily_return = 0.0

        cumulative_return = (equity - self.initial_capital) / self.initial_capital

        snapshot = DailySnapshot(
            date=str(current_date),
            equity=equity,
            cash=self.cash,
            open_positions=len(self.get_open_positions()),
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def run_daily_update(
        self,
        current_date: date,
        iv: float,
        rv: float,
        underlying_price: float,
    ) -> dict:
        """Run daily trading update.

        Args:
            current_date: Current date
            iv: Current implied volatility
            rv: Current realized volatility
            underlying_price: Current underlying price

        Returns:
            Summary of daily activity
        """
        logger.info(f"=== Paper Trading Update: {current_date} ===")
        logger.info(f"IV: {iv:.4f} ({iv*100:.2f}%), RV: {rv:.4f} ({rv*100:.2f}%)")
        logger.info(f"VRP: {(iv-rv):.4f} ({(iv-rv)*100:.2f}%)")

        # Check exits first
        closed_trades = self.check_exits(current_date, iv, rv)

        # Check for new positions
        new_positions = []
        should_open, position_type = self.should_open_position(iv, rv)
        if should_open:
            position = self.open_position(
                position_type, current_date, iv, rv, underlying_price
            )
            if position:
                new_positions.append(position)

        # Record snapshot
        snapshot = self.record_snapshot(current_date, iv, rv)

        # Save state
        self._save_state()

        summary = {
            "date": str(current_date),
            "iv": iv,
            "rv": rv,
            "vrp": iv - rv,
            "equity": snapshot.equity,
            "cash": snapshot.cash,
            "open_positions": snapshot.open_positions,
            "new_positions": len(new_positions),
            "closed_trades": len(closed_trades),
            "daily_return": snapshot.daily_return,
            "cumulative_return": snapshot.cumulative_return,
        }

        logger.info(f"Equity: ${snapshot.equity:,.2f}")
        logger.info(f"Cumulative Return: {snapshot.cumulative_return:.2%}")
        logger.info(f"Open Positions: {snapshot.open_positions}")

        return summary

    def get_performance_summary(self) -> dict:
        """Get overall performance summary."""
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]
        returns = [s.daily_return for s in self.snapshots if s.daily_return != 0]

        # Calculate metrics
        total_return = latest.cumulative_return
        n_days = len(self.snapshots)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        if returns:
            import numpy as np
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = annualized_return / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0

        # Trade stats
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            win_rate = len(wins) / len(self.trades)
            avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
            losses = [t for t in self.trades if t.pnl <= 0]
            avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
            profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "start_date": self.snapshots[0].date if self.snapshots else None,
            "end_date": latest.date,
            "trading_days": n_days,
            "initial_capital": self.initial_capital,
            "final_equity": latest.equity,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "open_positions": latest.open_positions,
            "cash": latest.cash,
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()

        df = pd.DataFrame([s.to_dict() for s in self.snapshots])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        return df

    def reset(self) -> None:
        """Reset paper trading state."""
        self.cash = self.initial_capital
        self.positions = []
        self.trades = []
        self.snapshots = []
        self.next_position_id = 1
        self.next_trade_id = 1
        self._save_state()
        logger.info("Paper trading state reset")
