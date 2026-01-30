"""Risk management for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.backtest.position import Position, PositionManager


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    max_position_size: float = 0.1  # Max position as fraction of capital
    max_portfolio_risk: float = 0.2  # Max total portfolio risk
    max_drawdown: float = 0.15  # Max acceptable drawdown
    stop_loss_pct: float = 0.5  # Stop loss as fraction of premium
    profit_target_pct: float = 0.5  # Profit target as fraction of premium
    max_delta: float = 0.3  # Max absolute portfolio delta
    max_vega: float = 0.1  # Max portfolio vega as fraction of capital
    max_concentration: float = 0.25  # Max concentration in single underlying


class RiskManager:
    """Risk manager for monitoring and controlling portfolio risk."""

    def __init__(
        self,
        position_manager: "PositionManager",
        limits: RiskLimits | None = None,
    ):
        """Initialize the risk manager.

        Args:
            position_manager: Position manager to monitor
            limits: Risk limits configuration
        """
        self.position_manager = position_manager
        self.limits = limits or RiskLimits()

        self.risk_history: list[dict] = []
        self.alerts: list[dict] = []

    def check_position_size(self, proposed_size: float) -> bool:
        """Check if proposed position size is within limits.

        Args:
            proposed_size: Proposed position notional value

        Returns:
            True if within limits
        """
        capital = self.position_manager.capital
        if proposed_size / capital > self.limits.max_position_size:
            self._log_alert("Position size exceeds limit", {
                "proposed_size": proposed_size,
                "limit": self.limits.max_position_size * capital,
            })
            return False
        return True

    def check_portfolio_risk(self) -> bool:
        """Check if total portfolio risk is within limits.

        Returns:
            True if within limits
        """
        total_exposure = sum(
            p.notional_value for p in self.position_manager.open_positions
        )
        capital = self.position_manager.capital

        if total_exposure / capital > self.limits.max_portfolio_risk:
            self._log_alert("Portfolio risk exceeds limit", {
                "total_exposure": total_exposure,
                "limit": self.limits.max_portfolio_risk * capital,
            })
            return False
        return True

    def check_drawdown(self) -> bool:
        """Check if current drawdown is within limits.

        Returns:
            True if within limits
        """
        equity_df = self.position_manager.get_equity_df()

        if equity_df.empty:
            return True

        equity = equity_df["equity"]
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak

        current_dd = drawdown.iloc[-1]

        if current_dd > self.limits.max_drawdown:
            self._log_alert("Drawdown exceeds limit", {
                "current_drawdown": current_dd,
                "limit": self.limits.max_drawdown,
            })
            return False
        return True

    def check_stop_loss(
        self,
        position: "Position",
        current_price: float,
    ) -> bool:
        """Check if position should be stopped out.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if stop loss triggered
        """
        # Calculate unrealized P&L as fraction of entry price
        unrealized_pnl = position.mark_to_market(current_price)
        entry_value = position.notional_value

        loss_pct = -unrealized_pnl / entry_value if entry_value > 0 else 0

        if loss_pct > self.limits.stop_loss_pct:
            self._log_alert("Stop loss triggered", {
                "position_id": position.position_id,
                "loss_pct": loss_pct,
                "limit": self.limits.stop_loss_pct,
            })
            return True
        return False

    def check_profit_target(
        self,
        position: "Position",
        current_price: float,
    ) -> bool:
        """Check if position has reached profit target.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if profit target reached
        """
        unrealized_pnl = position.mark_to_market(current_price)
        entry_value = position.notional_value

        profit_pct = unrealized_pnl / entry_value if entry_value > 0 else 0

        if profit_pct > self.limits.profit_target_pct:
            self._log_alert("Profit target reached", {
                "position_id": position.position_id,
                "profit_pct": profit_pct,
                "limit": self.limits.profit_target_pct,
            })
            return True
        return False

    def calculate_portfolio_greeks(
        self,
        prices: dict[int, dict],
    ) -> dict[str, float]:
        """Calculate aggregate portfolio Greeks.

        Args:
            prices: Dictionary mapping position_id to dict with price and greeks

        Returns:
            Dictionary with portfolio Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for position in self.position_manager.open_positions:
            if position.position_id not in prices:
                continue

            pos_data = prices[position.position_id]
            multiplier = position.quantity * (1 if position.position_type.value == 1 else -1)

            total_delta += pos_data.get("delta", 0) * multiplier * 100
            total_gamma += pos_data.get("gamma", 0) * multiplier * 100
            total_theta += pos_data.get("theta", 0) * multiplier * 100
            total_vega += pos_data.get("vega", 0) * multiplier * 100

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega,
        }

    def update_risk_metrics(
        self,
        current_date: date,
        prices: dict[int, dict] | None = None,
    ) -> dict:
        """Update and store risk metrics.

        Args:
            current_date: Current date
            prices: Current prices and greeks

        Returns:
            Current risk metrics
        """
        capital = self.position_manager.capital

        # Calculate exposure
        total_exposure = sum(
            p.notional_value for p in self.position_manager.open_positions
        )

        # Calculate Greeks if available
        greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        if prices:
            greeks = self.calculate_portfolio_greeks(prices)

        # Calculate drawdown
        equity_df = self.position_manager.get_equity_df()
        current_dd = 0.0
        if not equity_df.empty:
            equity = equity_df["equity"]
            peak = equity.expanding().max()
            drawdown = (peak - equity) / peak
            current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        metrics = {
            "date": current_date,
            "capital": capital,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / capital if capital > 0 else 0,
            "num_positions": self.position_manager.num_open_positions,
            "drawdown": current_dd,
            **greeks,
        }

        self.risk_history.append(metrics)
        return metrics

    def get_risk_report(self) -> pd.DataFrame:
        """Get risk history as DataFrame.

        Returns:
            DataFrame with risk metrics over time
        """
        if not self.risk_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.risk_history)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_alerts(self) -> pd.DataFrame:
        """Get all risk alerts.

        Returns:
            DataFrame with alerts
        """
        if not self.alerts:
            return pd.DataFrame()

        return pd.DataFrame(self.alerts)

    def _log_alert(self, message: str, details: dict) -> None:
        """Log a risk alert.

        Args:
            message: Alert message
            details: Alert details
        """
        alert = {
            "message": message,
            **details,
        }
        self.alerts.append(alert)
        logger.warning(f"Risk Alert: {message} - {details}")

    def can_open_position(
        self,
        proposed_size: float,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Args:
            proposed_size: Proposed position notional value

        Returns:
            Tuple of (can_open, reason)
        """
        # Check position count
        if not self.position_manager.can_open_position():
            return False, "Maximum positions reached"

        # Check position size
        if not self.check_position_size(proposed_size):
            return False, "Position size exceeds limit"

        # Check portfolio risk
        if not self.check_portfolio_risk():
            return False, "Portfolio risk exceeds limit"

        # Check drawdown
        if not self.check_drawdown():
            return False, "Drawdown exceeds limit"

        return True, "OK"

    def reset(self) -> None:
        """Reset the risk manager."""
        self.risk_history = []
        self.alerts = []
