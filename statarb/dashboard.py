"""Dashboard visualizations for data coverage and PnL/positions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class CoverageSummary:
    symbols: List[str]
    first_dates: pd.Series
    last_dates: pd.Series
    missing_pct: pd.Series
    missing_by_date: pd.Series


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_coverage(prices: pd.DataFrame) -> CoverageSummary:
    if prices.empty:
        raise ValueError("prices is empty")
    first_dates = prices.apply(lambda s: s.first_valid_index())
    last_dates = prices.apply(lambda s: s.last_valid_index())
    missing_pct = prices.isna().mean()
    missing_by_date = prices.isna().mean(axis=1)
    symbols = list(prices.columns)
    return CoverageSummary(
        symbols=symbols,
        first_dates=first_dates,
        last_dates=last_dates,
        missing_pct=missing_pct,
        missing_by_date=missing_by_date,
    )


def plot_data_coverage(
    prices: pd.DataFrame,
    output_dir: str,
    top_missing: int = 30,
) -> None:
    """Create data coverage dashboard plots."""
    summary = summarize_coverage(prices)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: listing start dates per symbol
    fig, ax = plt.subplots(figsize=(12, max(4, len(summary.symbols) * 0.12)))
    df_dates = pd.DataFrame(
        {
            "symbol": summary.symbols,
            "first_date": summary.first_dates.values,
            "last_date": summary.last_dates.values,
        }
    ).sort_values("first_date")
    span_days = (df_dates["last_date"] - df_dates["first_date"]).dt.days
    ax.barh(df_dates["symbol"], span_days)
    ax.set_title("Data Coverage Window per Symbol")
    ax.set_xlabel("Time Span (Days)")
    ax.set_ylabel("Symbol")
    fig.tight_layout()
    out_path = out_dir / "coverage_window.png"
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Plot 2: missing pct by symbol
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_sorted = summary.missing_pct.sort_values(ascending=False).head(top_missing)
    ax.bar(missing_sorted.index, missing_sorted.values)
    ax.set_title(f"Top {len(missing_sorted)} Symbols by Missing %")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Symbol")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    out_path = out_dir / "missing_by_symbol.png"
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Plot 3: missing pct by date
    fig, ax = plt.subplots(figsize=(12, 4))
    summary.missing_by_date.plot(ax=ax)
    ax.set_title("Missing Data % Over Time")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path = out_dir / "missing_by_date.png"
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pnl_and_positions(
    equity: pd.Series,
    weights: pd.DataFrame,
    output_dir: str,
    top_positions: int = 15,
) -> None:
    """Create PnL and positions dashboard plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: equity curve
    fig, ax = plt.subplots(figsize=(12, 4))
    equity.plot(ax=ax)
    ax.set_title("Equity Curve (Walk-Forward Backtest)")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path = out_dir / "equity_curve.png"
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Plot 2: rolling PnL (daily returns)
    returns = equity.pct_change().fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    returns.rolling(21).sum().plot(ax=ax)
    ax.set_title("Rolling 21D Return")
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path = out_dir / "rolling_21d_return.png"
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Plot 3: top position magnitudes over time
    if not weights.empty:
        avg_abs = weights.abs().mean().sort_values(ascending=False)
        top = avg_abs.head(top_positions).index
        fig, ax = plt.subplots(figsize=(12, 6))
        weights[top].plot(ax=ax)
        ax.set_title(f"Top {len(top)} Positions Over Time (Weights)")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Date")
        fig.tight_layout()
        out_path = out_dir / "top_positions.png"
        _ensure_parent(out_path)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def save_coverage_csv(prices: pd.DataFrame, output_path: str) -> None:
    summary = summarize_coverage(prices)
    df = pd.DataFrame(
        {
            "symbol": summary.symbols,
            "first_date": summary.first_dates.values,
            "last_date": summary.last_dates.values,
            "missing_pct": summary.missing_pct.values,
        }
    ).sort_values("missing_pct", ascending=False)
    out_path = Path(output_path)
    _ensure_parent(out_path)
    df.to_csv(out_path, index=False)
