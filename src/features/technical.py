"""Technical indicators for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalIndicators:
    """Calculator for technical indicators."""

    def __init__(self):
        """Initialize the technical indicators calculator."""
        pass

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average.

        Args:
            series: Price series
            window: Window size

        Returns:
            SMA series
        """
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average.

        Args:
            series: Price series
            span: EMA span

        Returns:
            EMA series
        """
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.

        Args:
            series: Price series
            window: RSI period

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR period

        Returns:
            ATR series
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / window, min_periods=window).mean()

        return atr

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Args:
            series: Price series
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def bollinger_width(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.Series:
        """Bollinger Band Width (normalized).

        Args:
            series: Price series
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            Bollinger width series
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(series, window, num_std)
        return (upper - lower) / middle

    @staticmethod
    def bollinger_pct_b(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.Series:
        """Bollinger %B (position within bands).

        Args:
            series: Price series
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            %B series (0 = lower band, 1 = upper band)
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(series, window, num_std)
        return (series - lower) / (upper - lower)

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence.

        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K period
            d_window: %D smoothing period

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_window).mean()

        return k, d

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Williams %R.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Lookback period

        Returns:
            Williams %R series (-100 to 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: CCI period

        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (volume * direction).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Volume Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def returns(series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate returns.

        Args:
            series: Price series
            periods: Number of periods

        Returns:
            Returns series
        """
        return series.pct_change(periods=periods)

    @staticmethod
    def log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate log returns.

        Args:
            series: Price series
            periods: Number of periods

        Returns:
            Log returns series
        """
        return np.log(series / series.shift(periods))

    @staticmethod
    def momentum(series: pd.Series, window: int = 10) -> pd.Series:
        """Price momentum.

        Args:
            series: Price series
            window: Lookback period

        Returns:
            Momentum series
        """
        return series.diff(window)

    @staticmethod
    def rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change (ROC).

        Args:
            series: Price series
            window: Lookback period

        Returns:
            ROC series (percentage)
        """
        return 100 * (series - series.shift(window)) / series.shift(window)

    def compute_all(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Compute all technical indicators.

        Args:
            df: DataFrame with OHLCV columns
            price_col: Column to use for price-based indicators

        Returns:
            DataFrame with all indicators
        """
        result = df.copy()

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            result[f"sma_{window}"] = self.sma(df[price_col], window)
            result[f"ema_{window}"] = self.ema(df[price_col], window)

        # Price relative to moving averages
        for window in [20, 50, 200]:
            result[f"price_to_sma_{window}"] = df[price_col] / result[f"sma_{window}"]

        # RSI
        for window in [7, 14, 21]:
            result[f"rsi_{window}"] = self.rsi(df[price_col], window)

        # ATR
        for window in [7, 14, 21]:
            result[f"atr_{window}"] = self.atr(df["high"], df["low"], df["close"], window)
            result[f"atr_pct_{window}"] = result[f"atr_{window}"] / df[price_col]

        # Bollinger Bands
        for window in [10, 20]:
            result[f"bb_width_{window}"] = self.bollinger_width(df[price_col], window)
            result[f"bb_pct_b_{window}"] = self.bollinger_pct_b(df[price_col], window)

        # MACD
        macd_line, signal_line, histogram = self.macd(df[price_col])
        result["macd_line"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = histogram

        # Stochastic
        k, d = self.stochastic(df["high"], df["low"], df["close"])
        result["stoch_k"] = k
        result["stoch_d"] = d

        # Williams %R
        result["williams_r"] = self.williams_r(df["high"], df["low"], df["close"])

        # CCI
        result["cci"] = self.cci(df["high"], df["low"], df["close"])

        # Volume indicators
        if "volume" in df.columns:
            result["obv"] = self.obv(df["close"], df["volume"])
            result["volume_sma_20"] = self.sma(df["volume"], 20)
            result["volume_ratio"] = df["volume"] / result["volume_sma_20"]

        # Returns
        for period in [1, 5, 10, 21]:
            result[f"return_{period}d"] = self.returns(df[price_col], period)
            result[f"log_return_{period}d"] = self.log_returns(df[price_col], period)

        # Momentum
        for window in [5, 10, 21]:
            result[f"momentum_{window}"] = self.momentum(df[price_col], window)
            result[f"roc_{window}"] = self.rate_of_change(df[price_col], window)

        logger.info(f"Computed {len(result.columns) - len(df.columns)} technical indicators")
        return result
