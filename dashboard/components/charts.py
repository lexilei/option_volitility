"""Reusable chart components for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    height: int = 500,
) -> go.Figure:
    """Create a candlestick chart.

    Args:
        df: DataFrame with OHLC columns
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    ])

    fig.update_layout(
        title=title,
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_white",
        height=height,
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_line_chart(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Line Chart",
    height: int = 400,
    y_title: str = "Value",
) -> go.Figure:
    """Create a multi-line chart.

    Args:
        df: DataFrame with data
        columns: Columns to plot
        title: Chart title
        height: Chart height
        y_title: Y-axis title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )

    return fig


def create_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    height: int = 400,
) -> go.Figure:
    """Create an equity curve chart.

    Args:
        equity: Equity series
        benchmark: Optional benchmark series
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity,
        mode="lines",
        name="Strategy",
        line=dict(color="blue"),
    ))

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark,
            mode="lines",
            name="Benchmark",
            line=dict(color="gray", dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )

    return fig


def create_drawdown_chart(
    equity: pd.Series,
    title: str = "Drawdown",
    height: int = 300,
) -> go.Figure:
    """Create a drawdown chart.

    Args:
        equity: Equity series
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    peak = equity.expanding().max()
    drawdown = (peak - equity) / peak

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color="red"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=height,
    )

    return fig


def create_returns_histogram(
    returns: pd.Series,
    title: str = "Returns Distribution",
    height: int = 400,
    nbins: int = 50,
) -> go.Figure:
    """Create a returns histogram.

    Args:
        returns: Returns series
        title: Chart title
        height: Chart height
        nbins: Number of bins

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=nbins,
        name="Returns",
    ))

    # Add normal distribution overlay
    mean = returns.mean() * 100
    std = returns.std() * 100
    x_range = np.linspace(mean - 4 * std, mean + 4 * std, 100)
    y_normal = len(returns) * (returns.max() - returns.min()) / nbins * \
               np.exp(-0.5 * ((x_range - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_normal,
        mode="lines",
        name="Normal",
        line=dict(color="red", dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=height,
    )

    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Correlation Matrix",
    height: int = 600,
) -> go.Figure:
    """Create a correlation heatmap.

    Args:
        df: DataFrame with features
        columns: Columns to include (None for all)
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    if columns is None:
        columns = list(df.columns)

    corr = df[columns].corr()

    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        x=columns,
        y=columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
    )

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "Bar Chart",
    height: int = 400,
    color: str | None = None,
) -> go.Figure:
    """Create a bar chart.

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        height: Chart height
        color: Column for color coding

    Returns:
        Plotly figure
    """
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
    )

    fig.update_layout(
        template="plotly_white",
        height=height,
    )

    return fig


def create_monthly_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns",
    height: int = 400,
) -> go.Figure:
    """Create a monthly returns heatmap.

    Args:
        returns: Daily returns series
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    }).pivot(index="year", columns="month", values="return")

    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = px.imshow(
        pivot * 100,
        labels=dict(color="Return (%)"),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
    )

    return fig


def create_feature_importance_chart(
    importance: pd.Series,
    title: str = "Feature Importance",
    height: int = 400,
    top_n: int = 20,
) -> go.Figure:
    """Create a feature importance chart.

    Args:
        importance: Series with feature importance
        title: Chart title
        height: Chart height
        top_n: Number of top features to show

    Returns:
        Plotly figure
    """
    importance = importance.sort_values(ascending=True).tail(top_n)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=height,
    )

    return fig


def create_subplots_chart(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Multi-Panel Chart",
    height_per_row: int = 200,
) -> go.Figure:
    """Create a multi-panel chart with subplots.

    Args:
        df: DataFrame with data
        columns: Columns to plot (one per panel)
        title: Overall chart title
        height_per_row: Height per subplot row

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=len(columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=columns,
    )

    for i, col in enumerate(columns, 1):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col, mode="lines"),
                row=i,
                col=1,
            )

    fig.update_layout(
        title=title,
        height=height_per_row * len(columns),
        template="plotly_white",
        showlegend=False,
    )

    return fig
