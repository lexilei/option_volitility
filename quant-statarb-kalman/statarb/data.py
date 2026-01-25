"""Data download/clean/align helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass
class PriceFetchResult:
    prices: pd.DataFrame
    missing_report: pd.Series


def fetch_prices(symbols: Iterable[str], start: str, end: str, source: str) -> pd.DataFrame:
    """Fetch adjusted close prices into a wide DataFrame [date x symbol].

    Source is a string identifier (e.g., "yfinance", "csv").
    """
    symbols = list(symbols)
    if not symbols:
        raise ValueError("symbols is empty")
    if source != "yfinance":
        raise NotImplementedError("Implement price download for source=%s" % source)
    df = yf.download(
        tickers=" ".join(symbols),
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        progress=False,
    )
    if df.empty:
        raise ValueError("No data returned from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)) or ("Adj Close" in df.columns.get_level_values(1)):
            try:
                df = df["Adj Close"]
            except KeyError:
                df = df.xs("Adj Close", axis=1, level=0, drop_level=True)
        else:
            df = df["Close"]
    else:
        df = df.rename(columns={"Adj Close": "adj_close", "Close": "close"})
        df = df[["adj_close"]] if "adj_close" in df.columns else df[["close"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def align_calendar(df: pd.DataFrame, max_missing_pct: float = 0.05) -> PriceFetchResult:
    """Align to a common calendar and drop symbols with excessive missing data."""
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    missing = df.isna().mean().sort_values(ascending=False)
    keep = missing[missing <= max_missing_pct].index
    aligned = df.loc[:, keep].dropna(how="any")
    return PriceFetchResult(prices=aligned, missing_report=missing)


def save_prices_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist prices to parquet."""
    df.to_parquet(path)
