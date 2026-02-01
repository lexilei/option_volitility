"""Trading strategy implementation for volatility selling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.models.base import BaseVolModel


class SignalType(Enum):
    """Trading signal types."""

    SELL_VOL = -1  # Sell volatility (short straddle/strangle)
    NEUTRAL = 0  # No position
    BUY_VOL = 1  # Buy volatility (long straddle/strangle)


@dataclass
class Signal:
    """A trading signal."""

    date: date
    signal_type: SignalType
    strength: float  # Signal strength (0 to 1)
    iv: float
    predicted_rv: float
    vrp: float  # Volatility risk premium
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """A completed trade."""

    entry_date: date
    exit_date: date
    signal_type: SignalType
    entry_iv: float
    exit_rv: float
    pnl: float  # P&L as fraction of premium
    holding_days: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VolatilityStrategy:
    """Volatility selling strategy based on VRP predictions."""

    def __init__(
        self,
        model: "BaseVolModel",
        vrp_threshold: float = 0.02,
        position_holding_days: int = 21,
        use_signal_strength: bool = True,
        max_positions: int = 1,
    ):
        """Initialize the strategy.

        Args:
            model: Trained volatility prediction model
            vrp_threshold: Minimum VRP to enter a position
            position_holding_days: Days to hold a position
            use_signal_strength: Whether to scale position by signal strength
            max_positions: Maximum concurrent positions
        """
        self.model = model
        self.vrp_threshold = vrp_threshold
        self.position_holding_days = position_holding_days
        self.use_signal_strength = use_signal_strength
        self.max_positions = max_positions

        self.signals: list[Signal] = []
        self.trades: list[Trade] = []

    def generate_signals(
        self,
        X: pd.DataFrame,
        iv: pd.Series,
        rv: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Generate trading signals.

        Args:
            X: Features for prediction
            iv: Implied volatility series
            rv: Realized volatility series (for backtesting)

        Returns:
            DataFrame with signals
        """
        # Get model predictions
        predicted_rv = self.model.predict(X)
        predicted_rv = pd.Series(predicted_rv, index=X.index)

        # Calculate VRP
        vrp = iv - predicted_rv

        # Generate signals
        signals_data = []
        self.signals = []

        for i, (idx, vrp_val) in enumerate(vrp.items()):
            if pd.isna(vrp_val) or pd.isna(iv.loc[idx]):
                signal_type = SignalType.NEUTRAL
                strength = 0.0
            elif vrp_val > self.vrp_threshold:
                # High VRP: sell volatility
                signal_type = SignalType.SELL_VOL
                # Signal strength based on VRP magnitude
                strength = min(1.0, vrp_val / (self.vrp_threshold * 3))
            elif vrp_val < -self.vrp_threshold:
                # Negative VRP: potentially buy volatility
                signal_type = SignalType.BUY_VOL
                strength = min(1.0, abs(vrp_val) / (self.vrp_threshold * 3))
            else:
                signal_type = SignalType.NEUTRAL
                strength = 0.0

            signal = Signal(
                date=idx,
                signal_type=signal_type,
                strength=strength,
                iv=iv.loc[idx] if not pd.isna(iv.loc[idx]) else 0.0,
                predicted_rv=predicted_rv.loc[idx] if not pd.isna(predicted_rv.loc[idx]) else 0.0,
                vrp=vrp_val if not pd.isna(vrp_val) else 0.0,
            )
            self.signals.append(signal)

            signals_data.append({
                "date": idx,
                "signal": signal_type.value,
                "signal_name": signal_type.name,
                "strength": strength,
                "iv": iv.loc[idx] if not pd.isna(iv.loc[idx]) else np.nan,
                "predicted_rv": predicted_rv.loc[idx],
                "vrp": vrp_val,
            })

        if not signals_data:
            logger.info("Generated 0 signals: empty input data")
            return pd.DataFrame(columns=["signal", "signal_name", "strength", "iv", "predicted_rv", "vrp"])

        df = pd.DataFrame(signals_data).set_index("date")
        logger.info(
            f"Generated {len(df)} signals: "
            f"{(df['signal'] == -1).sum()} SELL, "
            f"{(df['signal'] == 1).sum()} BUY, "
            f"{(df['signal'] == 0).sum()} NEUTRAL"
        )
        return df

    def backtest(
        self,
        signals_df: pd.DataFrame,
        realized_rv: pd.Series,
        price_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run backtest on generated signals.

        Simplified backtest assuming:
        - Selling vol: profit = IV - RV (normalized)
        - Buying vol: profit = RV - IV (normalized)

        Args:
            signals_df: DataFrame with signals
            realized_rv: Actual realized volatility
            price_data: Optional price data for more realistic P&L

        Returns:
            DataFrame with backtest results
        """
        self.trades = []
        results = []

        # Align data
        common_idx = signals_df.index.intersection(realized_rv.index)
        signals_df = signals_df.loc[common_idx]
        realized_rv = realized_rv.loc[common_idx]

        i = 0
        while i < len(signals_df) - self.position_holding_days:
            row = signals_df.iloc[i]
            signal = SignalType(row["signal"])

            if signal != SignalType.NEUTRAL:
                entry_date = signals_df.index[i]
                exit_idx = min(i + self.position_holding_days, len(signals_df) - 1)
                exit_date = signals_df.index[exit_idx]

                entry_iv = row["iv"]
                exit_rv = realized_rv.iloc[exit_idx]

                # Calculate P&L
                if signal == SignalType.SELL_VOL:
                    # Selling vol: profit when IV > RV
                    pnl = (entry_iv - exit_rv) / entry_iv if entry_iv > 0 else 0
                else:
                    # Buying vol: profit when RV > IV
                    pnl = (exit_rv - entry_iv) / entry_iv if entry_iv > 0 else 0

                # Apply signal strength if enabled
                if self.use_signal_strength:
                    pnl *= row["strength"]

                trade = Trade(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    signal_type=signal,
                    entry_iv=entry_iv,
                    exit_rv=exit_rv,
                    pnl=pnl,
                    holding_days=self.position_holding_days,
                )
                self.trades.append(trade)

                results.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "signal": signal.name,
                    "entry_iv": entry_iv,
                    "exit_rv": exit_rv,
                    "pnl": pnl,
                    "holding_days": self.position_holding_days,
                })

                # Skip to after position exit
                i = exit_idx + 1
            else:
                i += 1

        if not results:
            logger.warning("No trades executed in backtest")
            return pd.DataFrame()

        trades_df = pd.DataFrame(results)
        trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()

        logger.info(
            f"Backtest complete: {len(trades_df)} trades, "
            f"total P&L: {trades_df['pnl'].sum():.2%}"
        )
        return trades_df

    def get_equity_curve(
        self,
        signals_df: pd.DataFrame,
        realized_rv: pd.Series,
        initial_capital: float = 100000,
    ) -> pd.DataFrame:
        """Generate equity curve from backtest.

        Args:
            signals_df: DataFrame with signals
            realized_rv: Actual realized volatility
            initial_capital: Starting capital

        Returns:
            DataFrame with daily equity values
        """
        trades_df = self.backtest(signals_df, realized_rv)

        if trades_df.empty:
            return pd.DataFrame()

        # Create daily equity curve
        all_dates = pd.date_range(
            start=signals_df.index.min(),
            end=signals_df.index.max(),
            freq="D",
        )

        equity = pd.Series(float(initial_capital), index=all_dates, dtype=float)

        for _, trade in trades_df.iterrows():
            exit_date = pd.Timestamp(trade["exit_date"])
            if exit_date in equity.index:
                # Apply P&L at exit
                pnl_amount = initial_capital * trade["pnl"] * 0.1  # Scale factor
                equity.loc[exit_date:] += pnl_amount

        equity_df = pd.DataFrame({"equity": equity})
        equity_df["returns"] = equity_df["equity"].pct_change()
        equity_df["cumulative_returns"] = (1 + equity_df["returns"]).cumprod() - 1

        return equity_df


class SimpleVRPStrategy(VolatilityStrategy):
    """Simplified VRP-based strategy without a model.

    Uses historical VRP as the signal instead of predicted VRP.
    """

    def __init__(
        self,
        vrp_threshold: float = 0.02,
        position_holding_days: int = 21,
        lookback_days: int = 21,
    ):
        """Initialize the simple strategy.

        Args:
            vrp_threshold: Minimum VRP to enter position
            position_holding_days: Days to hold position
            lookback_days: Lookback for historical VRP calculation
        """
        self.vrp_threshold = vrp_threshold
        self.position_holding_days = position_holding_days
        self.lookback_days = lookback_days
        self.signals: list[Signal] = []
        self.trades: list[Trade] = []

    def generate_signals(
        self,
        iv: pd.Series,
        rv: pd.Series,
    ) -> pd.DataFrame:
        """Generate signals based on historical VRP.

        Args:
            iv: Implied volatility series
            rv: Realized volatility series

        Returns:
            DataFrame with signals
        """
        # Calculate historical VRP
        vrp = iv - rv

        signals_data = []
        self.signals = []

        for idx in vrp.index:
            vrp_val = vrp.loc[idx]

            if pd.isna(vrp_val):
                signal_type = SignalType.NEUTRAL
                strength = 0.0
            elif vrp_val > self.vrp_threshold:
                signal_type = SignalType.SELL_VOL
                strength = min(1.0, vrp_val / (self.vrp_threshold * 3))
            elif vrp_val < -self.vrp_threshold:
                signal_type = SignalType.BUY_VOL
                strength = min(1.0, abs(vrp_val) / (self.vrp_threshold * 3))
            else:
                signal_type = SignalType.NEUTRAL
                strength = 0.0

            signal = Signal(
                date=idx,
                signal_type=signal_type,
                strength=strength,
                iv=iv.loc[idx] if not pd.isna(iv.loc[idx]) else 0.0,
                predicted_rv=rv.loc[idx] if not pd.isna(rv.loc[idx]) else 0.0,
                vrp=vrp_val if not pd.isna(vrp_val) else 0.0,
            )
            self.signals.append(signal)

            signals_data.append({
                "date": idx,
                "signal": signal_type.value,
                "signal_name": signal_type.name,
                "strength": strength,
                "iv": iv.loc[idx],
                "predicted_rv": rv.loc[idx],
                "vrp": vrp_val,
            })

        return pd.DataFrame(signals_data).set_index("date")
