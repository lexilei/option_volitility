"""Polygon.io API client for fetching market data."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from loguru import logger


class PolygonClient:
    """Client for Polygon.io API."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str, rate_limit_delay: float = 0.25):
        """Initialize the Polygon client.

        Args:
            api_key: Polygon.io API key
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a request to the Polygon API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"Requesting: {url}")

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: date | str | None = None,
        to_date: date | str | None = None,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Get aggregate bars for a symbol.

        Args:
            symbol: Stock ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date
            to_date: End date
            adjusted: Whether to adjust for splits
            limit: Maximum number of results

        Returns:
            DataFrame with OHLCV data
        """
        if from_date is None:
            from_date = date.today() - timedelta(days=365)
        if to_date is None:
            to_date = date.today()

        if isinstance(from_date, date):
            from_date = from_date.strftime("%Y-%m-%d")
        if isinstance(to_date, date):
            to_date = to_date.strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
        }

        data = self._request(endpoint, params)

        if data.get("resultsCount", 0) == 0:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        results = data.get("results", [])
        df = pd.DataFrame(results)

        # Rename columns to standard names
        column_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "num_trades",
        }
        df = df.rename(columns=column_map)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        df = df.set_index("date")

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def get_options_chain(
        self,
        underlying_symbol: str,
        expiration_date: date | str | None = None,
        contract_type: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        expired: bool = False,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get options contracts for an underlying symbol.

        Args:
            underlying_symbol: The underlying stock ticker
            expiration_date: Filter by expiration date
            contract_type: Filter by type (call, put)
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            expired: Include expired contracts
            limit: Maximum number of results

        Returns:
            DataFrame with options contracts
        """
        endpoint = "/v3/reference/options/contracts"
        params: dict[str, Any] = {
            "underlying_ticker": underlying_symbol,
            "expired": str(expired).lower(),
            "limit": limit,
        }

        if expiration_date:
            if isinstance(expiration_date, date):
                expiration_date = expiration_date.strftime("%Y-%m-%d")
            params["expiration_date"] = expiration_date

        if contract_type:
            params["contract_type"] = contract_type

        if strike_price_gte:
            params["strike_price.gte"] = strike_price_gte

        if strike_price_lte:
            params["strike_price.lte"] = strike_price_lte

        all_results = []
        next_url = None

        while True:
            if next_url:
                # Extract endpoint from next_url
                response = requests.get(f"{next_url}&apiKey={self.api_key}", timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                data = self._request(endpoint, params)

            results = data.get("results", [])
            all_results.extend(results)

            next_url = data.get("next_url")
            if not next_url or len(all_results) >= limit:
                break

            self._rate_limit()

        if not all_results:
            logger.warning(f"No options found for {underlying_symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        logger.info(f"Fetched {len(df)} options contracts for {underlying_symbol}")
        return df

    def get_option_snapshot(self, underlying_symbol: str) -> pd.DataFrame:
        """Get current snapshot of all options for an underlying.

        Args:
            underlying_symbol: The underlying stock ticker

        Returns:
            DataFrame with options snapshot data
        """
        endpoint = f"/v3/snapshot/options/{underlying_symbol}"
        params = {"limit": 250}

        all_results = []
        next_url = None

        while True:
            if next_url:
                response = requests.get(f"{next_url}&apiKey={self.api_key}", timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                data = self._request(endpoint, params)

            results = data.get("results", [])
            all_results.extend(results)

            next_url = data.get("next_url")
            if not next_url:
                break

            self._rate_limit()

        if not all_results:
            logger.warning(f"No options snapshot for {underlying_symbol}")
            return pd.DataFrame()

        # Flatten the nested structure
        flattened = []
        for item in all_results:
            flat = {
                "ticker": item.get("details", {}).get("ticker"),
                "contract_type": item.get("details", {}).get("contract_type"),
                "expiration_date": item.get("details", {}).get("expiration_date"),
                "strike_price": item.get("details", {}).get("strike_price"),
                "underlying_ticker": item.get("underlying_asset", {}).get("ticker"),
                "underlying_price": item.get("underlying_asset", {}).get("price"),
                "implied_volatility": item.get("implied_volatility"),
                "open_interest": item.get("open_interest"),
                "bid": item.get("last_quote", {}).get("bid"),
                "ask": item.get("last_quote", {}).get("ask"),
                "last_price": item.get("last_trade", {}).get("price"),
                "volume": item.get("day", {}).get("volume"),
                "greeks_delta": item.get("greeks", {}).get("delta"),
                "greeks_gamma": item.get("greeks", {}).get("gamma"),
                "greeks_theta": item.get("greeks", {}).get("theta"),
                "greeks_vega": item.get("greeks", {}).get("vega"),
            }
            flattened.append(flat)

        df = pd.DataFrame(flattened)
        logger.info(f"Fetched {len(df)} options in snapshot for {underlying_symbol}")
        return df

    def get_ticker_details(self, symbol: str) -> dict[str, Any]:
        """Get details about a ticker.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        data = self._request(endpoint)
        return data.get("results", {})

    def get_market_status(self) -> dict[str, Any]:
        """Get current market status.

        Returns:
            Dictionary with market status
        """
        endpoint = "/v1/marketstatus/now"
        return self._request(endpoint)

    def get_previous_close(self, symbol: str, adjusted: bool = True) -> dict[str, Any]:
        """Get the previous day's close for a symbol.

        Args:
            symbol: Stock ticker symbol
            adjusted: Whether to adjust for splits

        Returns:
            Dictionary with previous close data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        params = {"adjusted": str(adjusted).lower()}
        data = self._request(endpoint, params)

        results = data.get("results", [])
        if results:
            return results[0]
        return {}
