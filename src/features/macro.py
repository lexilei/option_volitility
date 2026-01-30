"""Macro and market-level features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class MacroFeatures:
    """Calculator for macro and market-level features."""

    def __init__(self):
        """Initialize the macro features calculator."""
        pass

    @staticmethod
    def vix_level(vix_data: pd.DataFrame) -> pd.Series:
        """Get VIX closing level.

        Args:
            vix_data: DataFrame with VIX OHLC data

        Returns:
            VIX close series
        """
        return vix_data["close"]

    @staticmethod
    def vix_term_structure(
        vix_spot: pd.Series,
        vix_futures: pd.Series | None = None,
    ) -> pd.Series:
        """Calculate VIX term structure slope.

        Args:
            vix_spot: VIX spot prices
            vix_futures: VIX futures prices (optional)

        Returns:
            Term structure slope (positive = contango, negative = backwardation)
        """
        if vix_futures is None:
            # Approximate using VIX vs its moving average
            vix_ma = vix_spot.rolling(window=21).mean()
            return (vix_ma - vix_spot) / vix_spot
        return (vix_futures - vix_spot) / vix_spot

    @staticmethod
    def vix_percentile(
        vix: pd.Series,
        lookback: int = 252,
    ) -> pd.Series:
        """Calculate VIX percentile rank.

        Args:
            vix: VIX series
            lookback: Lookback period

        Returns:
            VIX percentile (0-100)
        """

        def pct_rank(x: pd.Series) -> float:
            if len(x) < 2:
                return np.nan
            return (x.rank(pct=True).iloc[-1]) * 100

        return vix.rolling(window=lookback).apply(pct_rank, raw=False)

    @staticmethod
    def vix_change(vix: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate VIX change.

        Args:
            vix: VIX series
            periods: Number of periods

        Returns:
            VIX percentage change
        """
        return vix.pct_change(periods=periods)

    @staticmethod
    def vix_spike(
        vix: pd.Series,
        threshold: float = 0.15,
    ) -> pd.Series:
        """Detect VIX spikes.

        Args:
            vix: VIX series
            threshold: Spike threshold (percentage)

        Returns:
            Binary spike indicator
        """
        vix_change = vix.pct_change()
        return (vix_change > threshold).astype(int)

    @staticmethod
    def market_regime(
        price: pd.Series,
        sma_short: int = 50,
        sma_long: int = 200,
    ) -> pd.Series:
        """Determine market regime based on moving averages.

        Args:
            price: Price series
            sma_short: Short-term MA period
            sma_long: Long-term MA period

        Returns:
            Regime indicator: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        ma_short = price.rolling(window=sma_short).mean()
        ma_long = price.rolling(window=sma_long).mean()

        regime = pd.Series(0, index=price.index)
        regime[ma_short > ma_long] = 1
        regime[ma_short < ma_long] = -1

        return regime

    @staticmethod
    def market_trend_strength(
        price: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate trend strength using ADX-like measure.

        Args:
            price: Price series
            window: Lookback window

        Returns:
            Trend strength (0-100)
        """
        returns = price.pct_change()
        positive_moves = returns.where(returns > 0, 0)
        negative_moves = (-returns).where(returns < 0, 0)

        smoothed_pos = positive_moves.rolling(window=window).mean()
        smoothed_neg = negative_moves.rolling(window=window).mean()

        di_diff = (smoothed_pos - smoothed_neg).abs()
        di_sum = smoothed_pos + smoothed_neg

        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.rolling(window=window).mean()

        return adx

    @staticmethod
    def put_call_ratio_signal(
        put_volume: pd.Series,
        call_volume: pd.Series,
        window: int = 5,
    ) -> pd.Series:
        """Calculate put-call ratio.

        Args:
            put_volume: Put option volume
            call_volume: Call option volume
            window: Smoothing window

        Returns:
            Smoothed put-call ratio
        """
        pcr = put_volume / call_volume.replace(0, np.nan)
        return pcr.rolling(window=window).mean()

    @staticmethod
    def correlation_with_market(
        returns: pd.Series,
        market_returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """Calculate rolling correlation with market.

        Args:
            returns: Asset returns
            market_returns: Market returns (e.g., SPY)
            window: Rolling window

        Returns:
            Rolling correlation
        """
        return returns.rolling(window=window).corr(market_returns)

    @staticmethod
    def beta(
        returns: pd.Series,
        market_returns: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """Calculate rolling beta.

        Args:
            returns: Asset returns
            market_returns: Market returns
            window: Rolling window

        Returns:
            Rolling beta
        """
        covariance = returns.rolling(window=window).cov(market_returns)
        market_variance = market_returns.rolling(window=window).var()
        return covariance / market_variance.replace(0, np.nan)

    @staticmethod
    def day_of_week(index: pd.DatetimeIndex | pd.Index) -> pd.Series:
        """Get day of week feature.

        Args:
            index: DateTime index

        Returns:
            Day of week (0=Monday, 4=Friday)
        """
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        return pd.Series(index.dayofweek, index=index)

    @staticmethod
    def month_of_year(index: pd.DatetimeIndex | pd.Index) -> pd.Series:
        """Get month of year feature.

        Args:
            index: DateTime index

        Returns:
            Month (1-12)
        """
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        return pd.Series(index.month, index=index)

    @staticmethod
    def is_month_end(index: pd.DatetimeIndex | pd.Index) -> pd.Series:
        """Check if date is near month end.

        Args:
            index: DateTime index

        Returns:
            Binary indicator
        """
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        return pd.Series(index.day >= 25, index=index).astype(int)

    @staticmethod
    def is_quarter_end(index: pd.DatetimeIndex | pd.Index) -> pd.Series:
        """Check if date is near quarter end.

        Args:
            index: DateTime index

        Returns:
            Binary indicator
        """
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        quarter_end_months = [3, 6, 9, 12]
        is_qe = pd.Series(
            (index.month.isin(quarter_end_months)) & (index.day >= 25),
            index=index,
        )
        return is_qe.astype(int)

    @staticmethod
    def days_to_expiration_effect(dte: pd.Series) -> pd.Series:
        """Calculate days-to-expiration effect.

        Args:
            dte: Days to expiration

        Returns:
            DTE effect (higher near expiration)
        """
        return 1 / (1 + dte / 30)

    def compute_all(
        self,
        price_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute all macro features.

        Args:
            price_df: DataFrame with price data
            vix_df: DataFrame with VIX data (optional)
            market_df: DataFrame with market data like SPY (optional)

        Returns:
            DataFrame with all macro features
        """
        result = price_df.copy()

        # Date features
        result["day_of_week"] = self.day_of_week(result.index)
        result["month"] = self.month_of_year(result.index)
        result["is_month_end"] = self.is_month_end(result.index)
        result["is_quarter_end"] = self.is_quarter_end(result.index)

        # Market regime
        result["market_regime"] = self.market_regime(price_df["close"])
        result["trend_strength"] = self.market_trend_strength(price_df["close"])

        # VIX features
        if vix_df is not None and not vix_df.empty:
            # Align VIX data with price data
            vix_aligned = vix_df["close"].reindex(result.index, method="ffill")

            result["vix"] = vix_aligned
            result["vix_pct_change_1d"] = self.vix_change(vix_aligned, 1)
            result["vix_pct_change_5d"] = self.vix_change(vix_aligned, 5)
            result["vix_percentile"] = self.vix_percentile(vix_aligned)
            result["vix_term_structure"] = self.vix_term_structure(vix_aligned)
            result["vix_spike"] = self.vix_spike(vix_aligned)

        # Market correlation/beta
        if market_df is not None and not market_df.empty:
            asset_returns = price_df["close"].pct_change()
            market_returns = market_df["close"].pct_change().reindex(result.index, method="ffill")

            result["market_corr_21d"] = self.correlation_with_market(
                asset_returns, market_returns, 21
            )
            result["market_corr_63d"] = self.correlation_with_market(
                asset_returns, market_returns, 63
            )
            result["beta_63d"] = self.beta(asset_returns, market_returns, 63)

        logger.info(f"Computed {len(result.columns) - len(price_df.columns)} macro features")
        return result
