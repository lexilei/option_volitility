"""Backtest Results page for the dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Backtest Results", page_icon="📈", layout="wide")

st.title("📈 Backtest Results")
st.markdown("View and analyze strategy backtest performance.")


@st.cache_data(ttl=3600)
def generate_sample_backtest():
    """Generate sample backtest data."""
    np.random.seed(42)

    # Equity curve
    n_days = 504
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    # Strategy returns (slightly positive drift)
    daily_returns = np.random.normal(0.0008, 0.012, n_days)
    equity = 100000 * np.cumprod(1 + daily_returns)

    equity_df = pd.DataFrame({
        "date": dates,
        "equity": equity,
        "returns": daily_returns,
    }).set_index("date")

    equity_df["cumulative_returns"] = (1 + equity_df["returns"]).cumprod() - 1

    # Calculate drawdown
    equity_df["peak"] = equity_df["equity"].expanding().max()
    equity_df["drawdown"] = (equity_df["peak"] - equity_df["equity"]) / equity_df["peak"]

    # Trades
    n_trades = 35
    trade_dates = np.random.choice(dates[:-21], n_trades, replace=False)
    trade_dates = sorted(trade_dates)

    trades = []
    for i, entry_date in enumerate(trade_dates):
        exit_date = entry_date + pd.Timedelta(days=np.random.randint(14, 28))
        pnl = np.random.normal(800, 2500)
        signal = np.random.choice(["SELL_VOL", "BUY_VOL"], p=[0.7, 0.3])

        trades.append({
            "trade_id": i + 1,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "signal": signal,
            "entry_iv": np.random.uniform(0.15, 0.35),
            "exit_rv": np.random.uniform(0.10, 0.30),
            "pnl": pnl,
            "return_pct": pnl / 10000,
        })

    trades_df = pd.DataFrame(trades)
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()

    # Monthly returns
    monthly = equity_df["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    return equity_df, trades_df, monthly_df


def plot_equity_curve(equity_df: pd.DataFrame):
    """Plot equity curve with drawdown."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown"],
    )

    # Equity
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["equity"],
            name="Equity",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["drawdown"] * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def plot_returns_distribution(equity_df: pd.DataFrame):
    """Plot returns distribution."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Daily Returns Distribution", "Q-Q Plot"],
    )

    # Histogram
    returns = equity_df["returns"].dropna() * 100
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name="Returns"),
        row=1,
        col=1,
    )

    # Q-Q plot (simplified)
    from scipy import stats

    sorted_returns = np.sort(returns)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_returns,
            mode="markers",
            marker=dict(size=4),
            name="Q-Q",
        ),
        row=1,
        col=2,
    )

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[-3, 3],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Normal",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_monthly_heatmap(monthly_df: pd.DataFrame):
    """Plot monthly returns heatmap."""
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = px.imshow(
        pivot * 100,
        labels=dict(color="Return (%)"),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
    )

    fig.update_layout(
        title="Monthly Returns Heatmap",
        template="plotly_white",
        height=300,
    )

    return fig


def plot_trade_analysis(trades_df: pd.DataFrame):
    """Plot trade analysis."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["P&L Distribution", "Cumulative P&L"],
    )

    # P&L histogram
    fig.add_trace(
        go.Histogram(x=trades_df["pnl"], nbinsx=20, name="P&L"),
        row=1,
        col=1,
    )

    # Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=trades_df["entry_date"],
            y=trades_df["cumulative_pnl"],
            mode="lines+markers",
            name="Cumulative",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def main():
    # Load data
    equity_df, trades_df, monthly_df = generate_sample_backtest()

    # Sidebar
    with st.sidebar:
        st.header("Backtest Settings")

        strategy = st.selectbox(
            "Strategy",
            ["VRP Strategy (ML)", "Simple VRP", "Buy & Hold"],
        )

        st.markdown("---")

        initial_capital = st.number_input(
            "Initial Capital",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
        )

        st.markdown("---")
        st.markdown("### Strategy Parameters")
        vrp_threshold = st.slider("VRP Threshold", 0.01, 0.10, 0.03)
        holding_days = st.slider("Holding Period (days)", 7, 42, 21)

    # Calculate metrics
    total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
    n_years = len(equity_df) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    volatility = equity_df["returns"].std() * np.sqrt(252)
    sharpe = ann_return / volatility if volatility > 0 else 0
    max_dd = equity_df["drawdown"].max()
    calmar = ann_return / max_dd if max_dd > 0 else 0

    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    total_trades = len(trades_df)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Main content
    # Key metrics
    st.subheader("Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{total_return:.1%}")
        st.metric("Annualized Return", f"{ann_return:.1%}")

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Calmar Ratio", f"{calmar:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{max_dd:.1%}")
        st.metric("Volatility", f"{volatility:.1%}")

    with col4:
        st.metric("Total Trades", f"{total_trades}")
        st.metric("Win Rate", f"{win_rate:.1%}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Equity Curve",
        "Returns Analysis",
        "Trade Analysis",
        "Risk Metrics",
    ])

    with tab1:
        st.subheader("Equity Curve & Drawdown")
        st.plotly_chart(plot_equity_curve(equity_df), use_container_width=True)

        # Monthly returns heatmap
        st.plotly_chart(plot_monthly_heatmap(monthly_df), use_container_width=True)

    with tab2:
        st.subheader("Returns Analysis")

        st.plotly_chart(plot_returns_distribution(equity_df), use_container_width=True)

        # Returns statistics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Daily Return Statistics")
            returns = equity_df["returns"].dropna()
            stats = {
                "Mean": f"{returns.mean() * 100:.3f}%",
                "Std Dev": f"{returns.std() * 100:.3f}%",
                "Skewness": f"{returns.skew():.3f}",
                "Kurtosis": f"{returns.kurtosis():.3f}",
                "Min": f"{returns.min() * 100:.2f}%",
                "Max": f"{returns.max() * 100:.2f}%",
            }
            st.dataframe(pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]))

        with col2:
            st.markdown("### Rolling Sharpe (252-day)")
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode="lines",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                template="plotly_white",
                height=300,
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Trade Analysis")

        st.plotly_chart(plot_trade_analysis(trades_df), use_container_width=True)

        # Trades table
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Winning Trades")
            winners = trades_df[trades_df["pnl"] > 0].sort_values("pnl", ascending=False)
            st.dataframe(winners.head(10), use_container_width=True)

        with col2:
            st.markdown("### Losing Trades")
            losers = trades_df[trades_df["pnl"] <= 0].sort_values("pnl", ascending=True)
            st.dataframe(losers.head(10), use_container_width=True)

        # Trade statistics by signal type
        st.markdown("### Performance by Signal Type")
        signal_stats = trades_df.groupby("signal").agg({
            "pnl": ["count", "sum", "mean", "std"],
            "return_pct": "mean",
        }).round(2)
        signal_stats.columns = ["Count", "Total P&L", "Avg P&L", "Std P&L", "Avg Return"]
        st.dataframe(signal_stats, use_container_width=True)

    with tab4:
        st.subheader("Risk Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Drawdown Analysis")

            dd_stats = {
                "Max Drawdown": f"{max_dd:.2%}",
                "Avg Drawdown": f"{equity_df['drawdown'].mean():.2%}",
                "Time in Drawdown": f"{(equity_df['drawdown'] > 0).mean():.1%}",
                "Max DD Duration": f"{(equity_df['drawdown'] > 0).astype(int).groupby((equity_df['drawdown'] == 0).cumsum()).sum().max()} days",
            }
            st.dataframe(pd.DataFrame(list(dd_stats.items()), columns=["Metric", "Value"]))

        with col2:
            st.markdown("### Risk-Adjusted Metrics")

            # Calculate additional metrics
            sortino = ann_return / (equity_df["returns"][equity_df["returns"] < 0].std() * np.sqrt(252))
            profit_factor = abs(trades_df[trades_df["pnl"] > 0]["pnl"].sum() / trades_df[trades_df["pnl"] < 0]["pnl"].sum())

            risk_metrics = {
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Sortino Ratio": f"{sortino:.2f}",
                "Calmar Ratio": f"{calmar:.2f}",
                "Profit Factor": f"{profit_factor:.2f}",
            }
            st.dataframe(pd.DataFrame(list(risk_metrics.items()), columns=["Metric", "Value"]))

        # VaR analysis
        st.markdown("### Value at Risk (VaR)")

        returns = equity_df["returns"].dropna()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()

        var_metrics = {
            "VaR (95%)": f"{var_95 * 100:.2f}%",
            "VaR (99%)": f"{var_99 * 100:.2f}%",
            "CVaR (95%)": f"{cvar_95 * 100:.2f}%",
        }

        col1, col2, col3 = st.columns(3)
        for i, (name, value) in enumerate(var_metrics.items()):
            with [col1, col2, col3][i]:
                st.metric(name, value)


if __name__ == "__main__":
    main()
