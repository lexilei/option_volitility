"""Unified data fetching interface."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from .massive_client import MassiveClient
from .storage import ParquetStorage

if TYPE_CHECKING:
    from pathlib import Path


class DataFetcher:
    """Unified interface for fetching and caching market data.

    Uses MassiveClient (formerly Polygon.io) for data fetching.
    """

    def __init__(
        self,
        api_key: str,
        data_dir: str | Path = "data",
        cache_enabled: bool = True,
    ):
        """Initialize the data fetcher.

        Args:
            api_key: Massive (Polygon.io) API key
            data_dir: Directory for data storage
            cache_enabled: Whether to cache data locally
        """
        self.client = MassiveClient(api_key)
        self.storage = ParquetStorage(data_dir)
        self.cache_enabled = cache_enabled

    def get_price_history(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (defaults to 2 years ago)
            end_date: End date (defaults to today)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=730)
        if end_date is None:
            end_date = date.today()

        cache_key = f"prices/{symbol}"

        # Try to load from cache
        if use_cache and self.cache_enabled:
            cached_df = self.storage.load(cache_key)
            if cached_df is not None:
                # Filter to requested date range
                cached_df.index = pd.to_datetime(cached_df.index).date
                mask = (cached_df.index >= start_date) & (cached_df.index <= end_date)
                filtered_df = cached_df[mask]

                # Check if we have all the data we need
                if len(filtered_df) > 0:
                    first_date = min(filtered_df.index)
                    last_date = max(filtered_df.index)

                    # If cache covers requested range, return it
                    if first_date <= start_date and last_date >= end_date - timedelta(days=5):
                        logger.info(f"Using cached price data for {symbol}")
                        return filtered_df

        # Fetch from API
        logger.info(f"Fetching price data for {symbol} from API")
        df = self.client.get_aggregates(symbol, from_date=start_date, to_date=end_date)

        if df.empty:
            return df

        # Cache the data
        if self.cache_enabled:
            # Load existing cache and merge
            existing = self.storage.load(cache_key)
            if existing is not None:
                existing.index = pd.to_datetime(existing.index).date
                df.index = pd.to_datetime(df.index).date
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()

            self.storage.save(df, cache_key)

        # Filter to requested range
        df.index = pd.to_datetime(df.index).date
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask]

    def get_options_chain_iv(
        self,
        symbol: str,
        expiration_date: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get options chain with IV data.

        Args:
            symbol: Underlying stock ticker
            expiration_date: Filter by expiration date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with options including IV and Greeks
        """
        cache_key = f"options_iv/{symbol}/{date.today().strftime('%Y%m%d')}"

        # Try cache (only use same-day cache for IV data)
        if use_cache and self.cache_enabled:
            cached_df = self.storage.load(cache_key)
            if cached_df is not None:
                logger.info(f"Using cached options IV for {symbol}")
                return cached_df

        # Fetch from API
        logger.info(f"Fetching options chain with IV for {symbol}")
        df = self.client.get_options_chain_iv(symbol, expiration_date=expiration_date)

        if df.empty:
            return df

        # Cache
        if self.cache_enabled:
            self.storage.save(df, cache_key)

        return df

    def get_atm_iv(self, symbol: str, days_to_expiry: int = 30) -> float | None:
        """Get ATM implied volatility for a symbol.

        Args:
            symbol: Underlying stock ticker
            days_to_expiry: Target days to expiration

        Returns:
            ATM IV as decimal (e.g., 0.20 for 20%)
        """
        return self.client.get_atm_iv(symbol, days_to_expiry)

    def get_vix_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get VIX historical data.

        Args:
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with VIX data
        """
        # VIX may not be available directly, try I:VIX or VIX
        for ticker in ["I:VIX", "VIX", "VIXY"]:
            try:
                df = self.get_price_history(ticker, start_date, end_date, use_cache)
                if not df.empty:
                    return df
            except Exception:
                continue

        logger.warning("VIX data not available")
        return pd.DataFrame()

    def get_multiple_symbols(
        self,
        symbols: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Get price history for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.get_price_history(symbol, start_date, end_date, use_cache)
                result[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                result[symbol] = pd.DataFrame()

        return result

    def refresh_cache(self, symbol: str, data_type: str = "prices") -> None:
        """Force refresh cached data for a symbol.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data to refresh (prices, options_iv)
        """
        cache_key = f"{data_type}/{symbol}"
        self.storage.delete(cache_key)

        if data_type == "prices":
            self.get_price_history(symbol, use_cache=False)
        elif data_type == "options_iv":
            self.get_options_chain_iv(symbol, use_cache=False)
