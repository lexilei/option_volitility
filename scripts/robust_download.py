#!/usr/bin/env python3
"""Robust price download with detailed diagnostics."""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


def download_single(symbol: str, start: str, end: str) -> tuple[pd.Series | None, str]:
    """Download a single symbol. Returns (series, status)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            return None, "NO_DATA"

        # Check first available date
        first_date = df.index[0].strftime("%Y-%m-%d")

        series = df["Close"]
        series.name = symbol
        series.index = pd.to_datetime(series.index).tz_localize(None)

        return series, f"OK (from {first_date})"
    except Exception as e:
        return None, f"ERROR: {str(e)[:50]}"


def main():
    # Load config
    config_path = Path("configs/universe.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    symbols = config["universe"]["symbols"]
    start = config["prices"]["start"]
    end = config["prices"]["end"]

    print(f"Downloading {len(symbols)} symbols from {start} to {end}")
    print("=" * 60)

    results = {}
    failed = []
    late_start = []  # Symbols that start after 2018

    start_dt = datetime.strptime(start, "%Y-%m-%d")

    for i, sym in enumerate(symbols):
        series, status = download_single(sym, start, end)

        if series is not None:
            results[sym] = series

            # Check if data starts significantly after requested start
            first_date = series.index[0]
            if first_date > pd.Timestamp(start) + pd.Timedelta(days=30):
                late_start.append((sym, first_date.strftime("%Y-%m-%d")))
                print(f"[{i+1:3d}/{len(symbols)}] {sym:6s} - {status} ⚠️  LATE START")
            else:
                print(f"[{i+1:3d}/{len(symbols)}] {sym:6s} - {status}")
        else:
            failed.append((sym, status))
            print(f"[{i+1:3d}/{len(symbols)}] {sym:6s} - {status} ❌")

        # Small delay to avoid rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"SUCCESS: {len(results)}/{len(symbols)}")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for sym, status in failed:
            print(f"  {sym}: {status}")

    if late_start:
        print(f"\nLATE START - no 2018 data ({len(late_start)}):")
        for sym, first_date in sorted(late_start, key=lambda x: x[1]):
            print(f"  {sym}: first data {first_date}")

    # Build DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.sort_index()

        # Report missing data
        print("\n" + "=" * 60)
        print("MISSING DATA REPORT:")
        missing = df.isna().mean().sort_values(ascending=False)
        high_missing = missing[missing > 0.05]
        if len(high_missing) > 0:
            print(f"\nSymbols with >5% missing ({len(high_missing)}):")
            for sym, pct in high_missing.head(20).items():
                print(f"  {sym}: {pct*100:.1f}%")
            if len(high_missing) > 20:
                print(f"  ... and {len(high_missing) - 20} more")

        # Save
        output_path = Path("data/processed/prices.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        print(f"\nSaved to {output_path}")
        print(f"Shape: {df.shape[0]} days x {df.shape[1]} symbols")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

        # Also save diagnostics
        diag_path = Path("data/processed/download_diagnostics.csv")
        diag_df = pd.DataFrame({
            "symbol": list(results.keys()),
            "first_date": [results[s].index[0].strftime("%Y-%m-%d") for s in results],
            "last_date": [results[s].index[-1].strftime("%Y-%m-%d") for s in results],
            "missing_pct": [df[s].isna().mean() * 100 for s in results],
            "n_rows": [results[s].notna().sum() for s in results],
        })
        diag_df = diag_df.sort_values("missing_pct", ascending=False)
        diag_df.to_csv(diag_path, index=False)
        print(f"Diagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
