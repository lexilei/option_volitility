"""Data download/clean/align helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
import yfinance as yf


@dataclass
class PriceFetchResult:
    prices: pd.DataFrame
    missing_report: pd.Series


def _normalize_yf_download(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)) or ("Adj Close" in df.columns.get_level_values(1)):
            try:
                df = df["Adj Close"]
            except KeyError:
                df = df.xs("Adj Close", axis=1, level=0, drop_level=True)
        else:
            df = df["Close"]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df = df[[col]].rename(columns={col: symbols[0]})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_prices(symbols: Iterable[str], start: str, end: str, source: str) -> pd.DataFrame:
    """Fetch adjusted close prices into a wide DataFrame [date x symbol].

    Source is a string identifier (e.g., "yfinance", "csv").
    """
    symbols = list(symbols)
    if not symbols:
        raise ValueError("symbols is empty")
    if source != "yfinance":
        raise NotImplementedError("Implement price download for source=%s" % source)
    # yfinance can be flaky for large batches; download in chunks and merge.
    frames: List[pd.DataFrame] = []
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        df = yf.download(
            tickers=" ".join(chunk),
            start=start,
            end=end,
            auto_adjust=True,
            group_by="column",
            progress=False,
        )
        df = _normalize_yf_download(df, chunk)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError("No data returned from yfinance")
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    # Recover any missing symbols by downloading individually.
    missing = [s for s in symbols if s not in out.columns]
    if missing:
        recovered: List[pd.DataFrame] = []
        for sym in missing:
            df = yf.download(
                tickers=sym,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="column",
                progress=False,
            )
            df = _normalize_yf_download(df, [sym])
            if not df.empty and sym in df.columns:
                recovered.append(df)
        if recovered:
            out = pd.concat([out] + recovered, axis=1)
            out = out.loc[:, ~out.columns.duplicated()]

    out = out.sort_index()
    return out


def align_calendar(df: pd.DataFrame, max_missing_pct: float = 0.05) -> PriceFetchResult:
    """Align to a common calendar and drop symbols with excessive missing data.

    Keep the full date index to avoid dropping rows due to sparse symbols;
    forward-fill so pairwise tests can decide on local overlap.
    """
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    missing = df.isna().mean().sort_values(ascending=False)
    keep = missing[missing <= max_missing_pct].index
    aligned = df.loc[:, keep].ffill().dropna(how="all")
    return PriceFetchResult(prices=aligned, missing_report=missing)


def save_prices_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist prices to parquet."""
    df.to_parquet(path)
