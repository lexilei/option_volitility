"""Massive (formerly Polygon.io) API client for fetching market data."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

try:
    from massive import RESTClient
except ImportError:
    RESTClient = None


class MassiveClient:
    """Client for Massive API (formerly Polygon.io)."""

    def __init__(self, api_key: str):
        """Initialize the Massive client.

        Args:
            api_key: Massive API key
        """
        if RESTClient is None:
            raise ImportError("massive is not installed. Run: pip install massive")

        self.api_key = api_key
        self.client = RESTClient(api_key)

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
            timespan: Size of the time window (day, week, etc.)
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

        results = []
        for agg in self.client.list_aggs(
            symbol,
            multiplier,
            timespan,
            from_date,
            to_date,
            adjusted=adjusted,
            limit=limit,
        ):
            results.append({
                "timestamp": agg.timestamp,
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
                "vwap": agg.vwap,
                "num_trades": agg.transactions,
            })

        if not results:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        df = df.set_index("date")

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def get_options_chain_iv(
        self,
        underlying_symbol: str,
        expiration_date: str | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Get options chain with IV data.

        Args:
            underlying_symbol: The underlying stock ticker (e.g., 'SPY')
            expiration_date: Filter by expiration date (YYYY-MM-DD)
            limit: Maximum number of options to fetch

        Returns:
            DataFrame with options data including IV
        """
        params: dict[str, Any] = {"limit": 250}
        if expiration_date:
            params["expiration_date"] = expiration_date

        results = []
        count = 0
        for option in self.client.list_snapshot_options_chain(
            underlying_symbol,
            params=params,
        ):
            if count >= limit:
                break

            details = option.details
            greeks = option.greeks
            underlying = option.underlying_asset

            results.append({
                "ticker": details.ticker if details else None,
                "contract_type": details.contract_type if details else None,
                "expiration_date": details.expiration_date if details else None,
                "strike_price": details.strike_price if details else None,
                "implied_volatility": option.implied_volatility,
                "open_interest": option.open_interest,
                "volume": option.day.volume if option.day else None,
                "delta": greeks.delta if greeks else None,
                "gamma": greeks.gamma if greeks else None,
                "theta": greeks.theta if greeks else None,
                "vega": greeks.vega if greeks else None,
                "underlying_price": underlying.price if underlying else None,
                "bid": option.last_quote.bid if option.last_quote else None,
                "ask": option.last_quote.ask if option.last_quote else None,
                "last_price": option.last_trade.price if option.last_trade else None,
            })
            count += 1

        if not results:
            logger.warning(f"No options found for {underlying_symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} options for {underlying_symbol}")
        return df

    def get_atm_iv(self, underlying_symbol: str, days_to_expiry: int = 30) -> float | None:
        """Get ATM implied volatility for an underlying.

        Calculates the average IV of near-ATM options expiring around
        the specified number of days.

        Args:
            underlying_symbol: The underlying stock ticker
            days_to_expiry: Target days to expiration (default 30 for ~VIX)

        Returns:
            ATM implied volatility as a decimal (e.g., 0.20 for 20%)
        """
        # Get options chain
        df = self.get_options_chain_iv(underlying_symbol, limit=2000)

        if df.empty:
            return None

        # Get underlying price from options data
        underlying_price = df["underlying_price"].dropna()
        if underlying_price.empty:
            # Use mid-point of strike prices as approximation
            strikes = df["strike_price"].dropna()
            if strikes.empty:
                logger.warning(f"Could not determine underlying price for {underlying_symbol}")
                return None
            underlying_price = strikes.median()
        else:
            underlying_price = underlying_price.iloc[0]

        # Filter to valid IV
        df = df[df["implied_volatility"].notna() & (df["implied_volatility"] > 0)]

        if df.empty:
            return None

        # Parse expiration dates
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])
        df["days_to_expiry"] = (df["expiration_date"] - pd.Timestamp.today()).dt.days

        # Filter to near target expiry (within 7 days)
        target_min = days_to_expiry - 7
        target_max = days_to_expiry + 7
        df_filtered = df[(df["days_to_expiry"] >= target_min) & (df["days_to_expiry"] <= target_max)]

        if df_filtered.empty:
            # Fall back to closest expiry
            df["expiry_diff"] = abs(df["days_to_expiry"] - days_to_expiry)
            closest_expiry = df.loc[df["expiry_diff"].idxmin(), "expiration_date"]
            df_filtered = df[df["expiration_date"] == closest_expiry]

        # Filter to near-ATM (within 5% of underlying price)
        atm_range = underlying_price * 0.05
        df_atm = df_filtered[
            (df_filtered["strike_price"] >= underlying_price - atm_range)
            & (df_filtered["strike_price"] <= underlying_price + atm_range)
        ]

        if df_atm.empty:
            df_atm = df_filtered

        # Calculate average IV (convert from percentage to decimal if needed)
        avg_iv = df_atm["implied_volatility"].mean()

        # If IV > 1, it's likely in percentage form
        if avg_iv > 1:
            avg_iv = avg_iv / 100

        logger.info(f"ATM IV for {underlying_symbol}: {avg_iv:.4f} ({avg_iv*100:.2f}%)")
        return avg_iv

    def get_historical_iv(
        self,
        underlying_symbol: str,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
    ) -> pd.Series:
        """Get historical ATM IV by fetching current snapshot.

        Note: Massive API doesn't provide historical IV directly.
        This returns current IV as a single point.
        For historical IV, you would need to store daily snapshots.

        Args:
            underlying_symbol: The underlying stock ticker
            from_date: Not used (for API compatibility)
            to_date: Not used (for API compatibility)

        Returns:
            Series with current ATM IV
        """
        iv = self.get_atm_iv(underlying_symbol)

        if iv is None:
            return pd.Series(dtype=float)

        return pd.Series({pd.Timestamp.today().date(): iv}, name="iv")
