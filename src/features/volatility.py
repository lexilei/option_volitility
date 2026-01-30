"""Volatility calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from loguru import logger


class VolatilityCalculator:
    """Calculator for various volatility metrics."""

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, annualization_factor: int | None = None):
        """Initialize the calculator.

        Args:
            annualization_factor: Factor to annualize volatility (default: 252)
        """
        self.annualization_factor = annualization_factor or self.TRADING_DAYS_PER_YEAR

    def realized_volatility(
        self,
        prices: pd.Series,
        window: int = 21,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Calculate realized volatility using close-to-close returns.

        Args:
            prices: Series of close prices
            window: Rolling window size (trading days)
            min_periods: Minimum periods required

        Returns:
            Series of annualized realized volatility
        """
        if min_periods is None:
            min_periods = window // 2

        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window=window, min_periods=min_periods).std()
        rv_annualized = rv * np.sqrt(self.annualization_factor)

        return rv_annualized

    def realized_volatility_parkinson(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 21,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Calculate Parkinson volatility using high-low range.

        More efficient than close-to-close for intraday data.

        Args:
            high: Series of high prices
            low: Series of low prices
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of annualized Parkinson volatility
        """
        if min_periods is None:
            min_periods = window // 2

        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))
        variance = factor * log_hl.rolling(window=window, min_periods=min_periods).mean()
        volatility = np.sqrt(variance * self.annualization_factor)

        return volatility

    def realized_volatility_garman_klass(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Calculate Garman-Klass volatility.

        Uses OHLC data for more accurate estimates.

        Args:
            open_: Series of open prices
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of annualized Garman-Klass volatility
        """
        if min_periods is None:
            min_periods = window // 2

        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2

        variance = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        variance_rolling = variance.rolling(window=window, min_periods=min_periods).mean()
        volatility = np.sqrt(variance_rolling * self.annualization_factor)

        return volatility

    def realized_volatility_yang_zhang(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Calculate Yang-Zhang volatility.

        Most accurate estimator using OHLC and overnight jumps.

        Args:
            open_: Series of open prices
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of annualized Yang-Zhang volatility
        """
        if min_periods is None:
            min_periods = window // 2

        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        # Overnight volatility
        log_oc = np.log(open_ / close.shift(1))
        overnight_var = log_oc.rolling(window=window, min_periods=min_periods).var()

        # Open-to-close volatility
        log_co = np.log(close / open_)
        open_close_var = log_co.rolling(window=window, min_periods=min_periods).var()

        # Rogers-Satchell volatility
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        rs = log_ho * log_hc + log_lo * log_lc
        rs_var = rs.rolling(window=window, min_periods=min_periods).mean()

        variance = overnight_var + k * open_close_var + (1 - k) * rs_var
        volatility = np.sqrt(variance * self.annualization_factor)

        return volatility

    @staticmethod
    def black_scholes_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate Black-Scholes option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0)
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def implied_volatility(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.05,
        option_type: str = "call",
    ) -> float | None:
        """Calculate implied volatility using bisection method.

        Args:
            option_price: Market option price
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Implied volatility or None if cannot be calculated
        """
        if T <= 0:
            return None

        # Define the objective function
        def objective(sigma: float) -> float:
            return self.black_scholes_price(S, K, T, r, sigma, option_type) - option_price

        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except ValueError:
            # No solution found in the given range
            return None

    def implied_volatility_vectorized(
        self,
        option_prices: pd.Series,
        spot_prices: pd.Series,
        strike_prices: pd.Series,
        time_to_expiry: pd.Series,
        risk_free_rate: float = 0.05,
        option_types: pd.Series | None = None,
    ) -> pd.Series:
        """Calculate implied volatility for a series of options.

        Args:
            option_prices: Series of option prices
            spot_prices: Series of underlying spot prices
            strike_prices: Series of strike prices
            time_to_expiry: Series of time to expiration (years)
            risk_free_rate: Risk-free rate
            option_types: Series of option types ('call' or 'put')

        Returns:
            Series of implied volatilities
        """
        if option_types is None:
            option_types = pd.Series(["call"] * len(option_prices), index=option_prices.index)

        ivs = []
        for i in range(len(option_prices)):
            iv = self.implied_volatility(
                option_price=option_prices.iloc[i],
                S=spot_prices.iloc[i],
                K=strike_prices.iloc[i],
                T=time_to_expiry.iloc[i],
                r=risk_free_rate,
                option_type=option_types.iloc[i],
            )
            ivs.append(iv)

        return pd.Series(ivs, index=option_prices.index)

    def volatility_risk_premium(
        self,
        iv: pd.Series,
        rv: pd.Series,
    ) -> pd.Series:
        """Calculate volatility risk premium (IV - RV).

        Args:
            iv: Series of implied volatility
            rv: Series of realized volatility

        Returns:
            Series of volatility risk premium
        """
        return iv - rv

    def volatility_ratio(
        self,
        iv: pd.Series,
        rv: pd.Series,
    ) -> pd.Series:
        """Calculate volatility ratio (IV / RV).

        Args:
            iv: Series of implied volatility
            rv: Series of realized volatility

        Returns:
            Series of volatility ratio
        """
        return iv / rv.replace(0, np.nan)

    def volatility_percentile(
        self,
        volatility: pd.Series,
        lookback: int = 252,
    ) -> pd.Series:
        """Calculate volatility percentile over a lookback period.

        Args:
            volatility: Series of volatility
            lookback: Lookback period in days

        Returns:
            Series of percentile ranks (0-100)
        """

        def percentile_rank(x: pd.Series) -> float:
            if len(x) < 2:
                return np.nan
            return (x.rank(pct=True).iloc[-1]) * 100

        return volatility.rolling(window=lookback).apply(percentile_rank, raw=False)

    def volatility_z_score(
        self,
        volatility: pd.Series,
        lookback: int = 252,
    ) -> pd.Series:
        """Calculate volatility z-score over a lookback period.

        Args:
            volatility: Series of volatility
            lookback: Lookback period in days

        Returns:
            Series of z-scores
        """
        rolling_mean = volatility.rolling(window=lookback).mean()
        rolling_std = volatility.rolling(window=lookback).std()
        return (volatility - rolling_mean) / rolling_std.replace(0, np.nan)

    def compute_all_rv_measures(
        self,
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute all realized volatility measures.

        Args:
            df: DataFrame with OHLC columns
            windows: List of window sizes to compute

        Returns:
            DataFrame with all volatility measures
        """
        if windows is None:
            windows = [5, 10, 21, 63]

        result = df.copy()

        for window in windows:
            suffix = f"_{window}d"

            # Close-to-close RV
            result[f"rv_cc{suffix}"] = self.realized_volatility(
                df["close"], window=window
            )

            # Parkinson RV
            result[f"rv_parkinson{suffix}"] = self.realized_volatility_parkinson(
                df["high"], df["low"], window=window
            )

            # Garman-Klass RV
            result[f"rv_gk{suffix}"] = self.realized_volatility_garman_klass(
                df["open"], df["high"], df["low"], df["close"], window=window
            )

            # Yang-Zhang RV
            result[f"rv_yz{suffix}"] = self.realized_volatility_yang_zhang(
                df["open"], df["high"], df["low"], df["close"], window=window
            )

        logger.info(f"Computed {len(windows) * 4} RV measures")
        return result
