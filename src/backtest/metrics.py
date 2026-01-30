"""Performance metrics for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # in days

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float

    # Other
    trading_days: int
    exposure_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "cumulative_return": self.cumulative_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade": self.avg_trade,
            "trading_days": self.trading_days,
            "exposure_pct": self.exposure_pct,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for display."""
        metrics = self.to_dict()
        df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        return df

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annualized Return: {self.annualized_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Total Trades: {self.total_trades}\n"
            f")"
        )


def calculate_returns_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Calculate return-based metrics.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year

    Returns:
        Dictionary of metrics
    """
    if returns.empty or returns.isna().all():
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "cumulative_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

    # Clean returns
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "cumulative_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

    # Total and cumulative return
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # Annualized return
    n_years = len(returns) / trading_days_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(trading_days_per_year)

    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1

    # Sharpe ratio
    excess_returns = returns - daily_rf
    sharpe_ratio = (
        excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days_per_year)
        if excess_returns.std() > 0
        else 0
    )

    # Sortino ratio (uses only downside deviation)
    downside_returns = returns[returns < daily_rf]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (
        (returns.mean() - daily_rf) / downside_std * np.sqrt(trading_days_per_year)
        if downside_std > 0
        else 0
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "cumulative_return": total_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }


def calculate_drawdown_metrics(equity: pd.Series) -> dict[str, float]:
    """Calculate drawdown-based metrics.

    Args:
        equity: Series of equity values

    Returns:
        Dictionary of drawdown metrics
    """
    if equity.empty or equity.isna().all():
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration": 0,
        }

    equity = equity.dropna()

    if len(equity) == 0:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration": 0,
        }

    # Calculate drawdown series
    peak = equity.expanding().max()
    drawdown = (peak - equity) / peak

    max_drawdown = drawdown.max()
    avg_drawdown = drawdown.mean()

    # Max drawdown duration
    in_drawdown = drawdown > 0
    drawdown_periods = []
    current_period = 0

    for dd in in_drawdown:
        if dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0

    if current_period > 0:
        drawdown_periods.append(current_period)

    max_dd_duration = max(drawdown_periods) if drawdown_periods else 0

    return {
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "max_drawdown_duration": max_dd_duration,
    }


def calculate_trade_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    """Calculate trade-based metrics.

    Args:
        trades_df: DataFrame with trade history (must have 'pnl' column)

    Returns:
        Dictionary of trade metrics
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade": 0.0,
        }

    pnl = trades_df["pnl"]

    winning = pnl[pnl > 0]
    losing = pnl[pnl <= 0]

    total_trades = len(pnl)
    winning_trades = len(winning)
    losing_trades = len(losing)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    gross_profit = winning.sum() if len(winning) > 0 else 0
    gross_loss = abs(losing.sum()) if len(losing) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = winning.mean() if len(winning) > 0 else 0
    avg_loss = losing.mean() if len(losing) > 0 else 0
    largest_win = pnl.max()
    largest_loss = pnl.min()
    avg_trade = pnl.mean()

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_trade": avg_trade,
    }


def calculate_all_metrics(
    equity: pd.Series,
    trades_df: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate all performance metrics.

    Args:
        equity: Series of equity values
        trades_df: DataFrame with trade history
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    # Calculate returns from equity
    returns = equity.pct_change().dropna()

    # Return metrics
    return_metrics = calculate_returns_metrics(returns, risk_free_rate)

    # Drawdown metrics
    dd_metrics = calculate_drawdown_metrics(equity)

    # Trade metrics
    if trades_df is not None:
        trade_metrics = calculate_trade_metrics(trades_df)
    else:
        trade_metrics = calculate_trade_metrics(pd.DataFrame())

    # Calmar ratio
    calmar_ratio = (
        return_metrics["annualized_return"] / dd_metrics["max_drawdown"]
        if dd_metrics["max_drawdown"] > 0
        else 0
    )

    # Exposure (simplified)
    exposure_pct = (trade_metrics["total_trades"] * 21) / len(equity) if len(equity) > 0 else 0
    exposure_pct = min(1.0, exposure_pct)

    return PerformanceMetrics(
        total_return=return_metrics["total_return"],
        annualized_return=return_metrics["annualized_return"],
        cumulative_return=return_metrics["cumulative_return"],
        volatility=return_metrics["volatility"],
        max_drawdown=dd_metrics["max_drawdown"],
        avg_drawdown=dd_metrics["avg_drawdown"],
        max_drawdown_duration=dd_metrics["max_drawdown_duration"],
        sharpe_ratio=return_metrics["sharpe_ratio"],
        sortino_ratio=return_metrics["sortino_ratio"],
        calmar_ratio=calmar_ratio,
        total_trades=trade_metrics["total_trades"],
        winning_trades=trade_metrics["winning_trades"],
        losing_trades=trade_metrics["losing_trades"],
        win_rate=trade_metrics["win_rate"],
        profit_factor=trade_metrics["profit_factor"],
        avg_win=trade_metrics["avg_win"],
        avg_loss=trade_metrics["avg_loss"],
        largest_win=trade_metrics["largest_win"],
        largest_loss=trade_metrics["largest_loss"],
        avg_trade=trade_metrics["avg_trade"],
        trading_days=len(equity),
        exposure_pct=exposure_pct,
    )


def compare_strategies(
    results: dict[str, tuple[pd.Series, pd.DataFrame]],
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compare multiple strategy results.

    Args:
        results: Dictionary mapping strategy name to (equity, trades_df) tuple
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame comparing all strategies
    """
    comparison = {}

    for name, (equity, trades_df) in results.items():
        metrics = calculate_all_metrics(equity, trades_df, risk_free_rate)
        comparison[name] = metrics.to_dict()

    df = pd.DataFrame(comparison).T
    df.index.name = "Strategy"

    # Format certain columns as percentages
    pct_cols = [
        "total_return",
        "annualized_return",
        "cumulative_return",
        "volatility",
        "max_drawdown",
        "avg_drawdown",
        "win_rate",
        "exposure_pct",
    ]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}")

    return df
