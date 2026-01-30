"""Position management for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class PositionType(Enum):
    """Types of positions."""

    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Position:
    """A single position."""

    position_id: int
    entry_date: date
    position_type: PositionType
    entry_price: float
    quantity: int
    underlying_price: float
    strike: float | None = None
    expiration: date | None = None
    option_type: str | None = None  # 'call' or 'put'
    exit_date: date | None = None
    exit_price: float | None = None
    pnl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_date is None

    @property
    def notional_value(self) -> float:
        """Get notional value of position."""
        return abs(self.entry_price * self.quantity * 100)  # Options are 100 shares

    def close(self, exit_date: date, exit_price: float) -> float:
        """Close the position.

        Args:
            exit_date: Date of exit
            exit_price: Price at exit

        Returns:
            P&L of the position
        """
        self.exit_date = exit_date
        self.exit_price = exit_price

        # Calculate P&L
        if self.position_type == PositionType.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity * 100
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity * 100

        return self.pnl

    def mark_to_market(self, current_price: float) -> float:
        """Get unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if self.position_type == PositionType.LONG:
            return (current_price - self.entry_price) * self.quantity * 100
        else:
            return (self.entry_price - current_price) * self.quantity * 100


class PositionManager:
    """Manager for tracking and managing positions."""

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_size: float = 0.1,
        max_positions: int = 5,
    ):
        """Initialize the position manager.

        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.max_positions = max_positions

        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.position_counter = 0

        self.equity_history: list[dict] = []

    @property
    def open_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.is_open]

    @property
    def num_open_positions(self) -> int:
        """Get number of open positions."""
        return len(self.open_positions)

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return self.num_open_positions < self.max_positions

    def get_position_size(self, price: float) -> int:
        """Calculate position size based on risk rules.

        Args:
            price: Entry price

        Returns:
            Number of contracts
        """
        max_notional = self.capital * self.max_position_size
        # Each contract is 100 shares
        contracts = int(max_notional / (price * 100))
        return max(1, contracts)

    def open_position(
        self,
        entry_date: date,
        position_type: PositionType,
        entry_price: float,
        underlying_price: float,
        quantity: int | None = None,
        strike: float | None = None,
        expiration: date | None = None,
        option_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Position | None:
        """Open a new position.

        Args:
            entry_date: Date of entry
            position_type: Long or short
            entry_price: Entry price
            underlying_price: Price of underlying
            quantity: Number of contracts (auto-calculated if None)
            strike: Strike price (for options)
            expiration: Expiration date (for options)
            option_type: 'call' or 'put'
            metadata: Additional metadata

        Returns:
            The opened position or None if cannot open
        """
        if not self.can_open_position():
            logger.warning("Cannot open position: max positions reached")
            return None

        if quantity is None:
            quantity = self.get_position_size(entry_price)

        self.position_counter += 1

        position = Position(
            position_id=self.position_counter,
            entry_date=entry_date,
            position_type=position_type,
            entry_price=entry_price,
            quantity=quantity,
            underlying_price=underlying_price,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            metadata=metadata or {},
        )

        self.positions.append(position)
        logger.debug(
            f"Opened {position_type.name} position #{position.position_id}: "
            f"{quantity} contracts @ {entry_price:.2f}"
        )

        return position

    def close_position(
        self,
        position: Position,
        exit_date: date,
        exit_price: float,
    ) -> float:
        """Close a specific position.

        Args:
            position: Position to close
            exit_date: Date of exit
            exit_price: Exit price

        Returns:
            P&L of the closed position
        """
        pnl = position.close(exit_date, exit_price)
        self.capital += pnl
        self.closed_positions.append(position)

        logger.debug(
            f"Closed position #{position.position_id}: "
            f"P&L = ${pnl:.2f}"
        )

        return pnl

    def close_all_positions(self, exit_date: date, prices: dict[int, float]) -> float:
        """Close all open positions.

        Args:
            exit_date: Date of exit
            prices: Dictionary mapping position_id to exit price

        Returns:
            Total P&L
        """
        total_pnl = 0.0

        for position in self.open_positions:
            if position.position_id in prices:
                pnl = self.close_position(position, exit_date, prices[position.position_id])
                total_pnl += pnl

        return total_pnl

    def update_equity(self, current_date: date, prices: dict[int, float]) -> float:
        """Update equity based on current prices.

        Args:
            current_date: Current date
            prices: Dictionary mapping position_id to current price

        Returns:
            Current equity
        """
        unrealized_pnl = 0.0

        for position in self.open_positions:
            if position.position_id in prices:
                unrealized_pnl += position.mark_to_market(prices[position.position_id])

        equity = self.capital + unrealized_pnl

        self.equity_history.append({
            "date": current_date,
            "capital": self.capital,
            "unrealized_pnl": unrealized_pnl,
            "equity": equity,
            "num_positions": self.num_open_positions,
        })

        return equity

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity history as DataFrame.

        Returns:
            DataFrame with equity history
        """
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["returns"] = df["equity"].pct_change()
        df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1

        return df

    def get_trades_df(self) -> pd.DataFrame:
        """Get closed trades as DataFrame.

        Returns:
            DataFrame with trade history
        """
        if not self.closed_positions:
            return pd.DataFrame()

        records = []
        for p in self.closed_positions:
            records.append({
                "position_id": p.position_id,
                "entry_date": p.entry_date,
                "exit_date": p.exit_date,
                "position_type": p.position_type.name,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "quantity": p.quantity,
                "pnl": p.pnl,
                "return_pct": p.pnl / p.notional_value if p.notional_value > 0 else 0,
            })

        return pd.DataFrame(records)

    def get_summary(self) -> dict[str, Any]:
        """Get position manager summary.

        Returns:
            Dictionary with summary statistics
        """
        trades_df = self.get_trades_df()

        if trades_df.empty:
            return {
                "total_trades": 0,
                "open_positions": self.num_open_positions,
                "initial_capital": self.initial_capital,
                "current_capital": self.capital,
            }

        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] <= 0]

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            "total_pnl": trades_df["pnl"].sum(),
            "avg_pnl": trades_df["pnl"].mean(),
            "avg_winner": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loser": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_winner": trades_df["pnl"].max(),
            "largest_loser": trades_df["pnl"].min(),
            "open_positions": self.num_open_positions,
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
        }

    def reset(self) -> None:
        """Reset the position manager."""
        self.capital = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.position_counter = 0
        self.equity_history = []
