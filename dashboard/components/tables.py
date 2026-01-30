"""Reusable table components for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a value as currency."""
    return f"${value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas."""
    return f"{value:,.{decimals}f}"


def create_metrics_table(
    metrics: dict[str, float],
    format_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a formatted metrics table.

    Args:
        metrics: Dictionary of metric names and values
        format_dict: Dictionary specifying format for each metric
            ('pct' for percentage, 'currency' for currency, 'number' for plain number)

    Returns:
        Formatted DataFrame
    """
    if format_dict is None:
        format_dict = {}

    formatted = {}
    for name, value in metrics.items():
        fmt = format_dict.get(name, "number")
        if fmt == "pct":
            formatted[name] = format_percentage(value)
        elif fmt == "currency":
            formatted[name] = format_currency(value)
        else:
            formatted[name] = format_number(value)

    return pd.DataFrame(
        list(formatted.items()),
        columns=["Metric", "Value"],
    )


def display_performance_metrics(
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    total_trades: int,
    additional_metrics: dict[str, float] | None = None,
) -> None:
    """Display performance metrics in a grid.

    Args:
        total_return: Total return
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        win_rate: Win rate
        total_trades: Total number of trades
        additional_metrics: Additional metrics to display
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Return", format_percentage(total_return))

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    with col3:
        st.metric("Max Drawdown", format_percentage(max_drawdown))

    with col4:
        st.metric("Win Rate", format_percentage(win_rate))

    with col5:
        st.metric("Total Trades", str(total_trades))

    if additional_metrics:
        cols = st.columns(len(additional_metrics))
        for i, (name, value) in enumerate(additional_metrics.items()):
            with cols[i]:
                if isinstance(value, float):
                    if abs(value) < 10:
                        st.metric(name, f"{value:.2f}")
                    else:
                        st.metric(name, format_number(value, 0))
                else:
                    st.metric(name, str(value))


def create_model_comparison_table(
    results: dict[str, dict[str, float]],
    highlight_best: bool = True,
) -> pd.DataFrame:
    """Create a model comparison table.

    Args:
        results: Dictionary mapping model name to metrics dict
        highlight_best: Whether to highlight best values

    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame(results).T

    # Determine which columns should be minimized vs maximized
    minimize_cols = ["RMSE", "MAE", "MAPE", "Max Drawdown"]
    maximize_cols = ["R2", "Sharpe", "Sortino", "Calmar", "Win Rate"]

    if highlight_best:
        # This returns the styled dataframe
        styled = df.style

        for col in df.columns:
            if col in minimize_cols:
                styled = styled.highlight_min(subset=[col], color="lightgreen")
            elif col in maximize_cols:
                styled = styled.highlight_max(subset=[col], color="lightgreen")

        return styled

    return df


def create_trades_table(
    trades_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a formatted trades table.

    Args:
        trades_df: DataFrame with trade data
        columns: Columns to include (None for default)

    Returns:
        Formatted DataFrame
    """
    if columns is None:
        columns = [
            "entry_date",
            "exit_date",
            "signal",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
        ]

    df = trades_df[columns].copy()

    # Format columns
    if "pnl" in df.columns:
        df["pnl"] = df["pnl"].apply(lambda x: format_currency(x))

    if "return_pct" in df.columns:
        df["return_pct"] = df["return_pct"].apply(lambda x: format_percentage(x))

    if "entry_price" in df.columns:
        df["entry_price"] = df["entry_price"].apply(lambda x: format_currency(x))

    if "exit_price" in df.columns:
        df["exit_price"] = df["exit_price"].apply(lambda x: format_currency(x))

    return df


def create_summary_stats_table(
    data: pd.Series,
    name: str = "Value",
) -> pd.DataFrame:
    """Create a summary statistics table.

    Args:
        data: Data series to summarize
        name: Name for the column

    Returns:
        DataFrame with summary statistics
    """
    stats = {
        "Count": len(data),
        "Mean": data.mean(),
        "Std Dev": data.std(),
        "Min": data.min(),
        "25%": data.quantile(0.25),
        "50%": data.quantile(0.50),
        "75%": data.quantile(0.75),
        "Max": data.max(),
        "Skewness": data.skew(),
        "Kurtosis": data.kurtosis(),
    }

    df = pd.DataFrame(list(stats.items()), columns=["Statistic", name])
    df[name] = df[name].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))

    return df


def create_correlation_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a table of highly correlated features.

    Args:
        df: DataFrame with features
        columns: Columns to include
        threshold: Minimum absolute correlation to include

    Returns:
        DataFrame with correlation pairs
    """
    if columns is None:
        columns = list(df.columns)

    corr = df[columns].corr()

    pairs = []
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j:
                r = corr.loc[col1, col2]
                if abs(r) >= threshold:
                    pairs.append({
                        "Feature 1": col1,
                        "Feature 2": col2,
                        "Correlation": r,
                    })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result["Abs Correlation"] = result["Correlation"].abs()
        result = result.sort_values("Abs Correlation", ascending=False)
        result["Correlation"] = result["Correlation"].apply(lambda x: f"{x:.4f}")
        result = result.drop(columns=["Abs Correlation"])

    return result


def display_dataframe_with_pagination(
    df: pd.DataFrame,
    page_size: int = 20,
    key: str = "pagination",
) -> None:
    """Display a DataFrame with pagination.

    Args:
        df: DataFrame to display
        page_size: Rows per page
        key: Unique key for session state
    """
    n_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)

    if f"{key}_page" not in st.session_state:
        st.session_state[f"{key}_page"] = 0

    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("Previous", key=f"{key}_prev"):
            if st.session_state[f"{key}_page"] > 0:
                st.session_state[f"{key}_page"] -= 1

    with col2:
        st.write(f"Page {st.session_state[f'{key}_page'] + 1} of {n_pages}")

    with col3:
        if st.button("Next", key=f"{key}_next"):
            if st.session_state[f"{key}_page"] < n_pages - 1:
                st.session_state[f"{key}_page"] += 1

    # Display current page
    start_idx = st.session_state[f"{key}_page"] * page_size
    end_idx = start_idx + page_size
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
