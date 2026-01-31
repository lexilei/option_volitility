"""Paper Trading page for the dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Paper Trading", page_icon="💰", layout="wide")

st.title("💰 Paper Trading")
st.markdown("Live paper trading simulation with real market data.")


def load_trading_data(data_dir: str = "data") -> dict:
    """Load paper trading data from files."""
    trading_dir = Path(data_dir) / "paper_trading"

    data = {
        "state": None,
        "trades": [],
        "snapshots": [],
        "summary": None,
    }

    # Load state
    state_file = trading_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            data["state"] = json.load(f)

    # Load trades
    trades_file = trading_dir / "trades.json"
    if trades_file.exists():
        with open(trades_file) as f:
            data["trades"] = json.load(f)

    # Load snapshots
    snapshots_file = trading_dir / "snapshots.json"
    if snapshots_file.exists():
        with open(snapshots_file) as f:
            data["snapshots"] = json.load(f)

    # Load latest summary
    summary_file = trading_dir / "latest_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data["summary"] = json.load(f)

    return data


def plot_equity_curve(snapshots: list) -> go.Figure:
    """Plot equity curve with drawdowns."""
    if not snapshots:
        return None

    df = pd.DataFrame(snapshots)
    df["date"] = pd.to_datetime(df["date"])

    # Calculate drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.1)",
        ),
        row=1, col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.2)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def plot_returns_distribution(snapshots: list) -> go.Figure:
    """Plot daily returns distribution."""
    if not snapshots:
        return None

    df = pd.DataFrame(snapshots)
    returns = df["daily_return"].dropna() * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name="Daily Returns",
        marker_color="#3498db",
    ))

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # Add mean line
    mean_return = returns.mean()
    fig.add_vline(x=mean_return, line_dash="solid", line_color="#e74c3c",
                  annotation_text=f"Mean: {mean_return:.2f}%")

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
    )

    return fig


def plot_trades_pnl(trades: list) -> go.Figure:
    """Plot P&L by trade."""
    if not trades:
        return None

    df = pd.DataFrame(trades)

    colors = ["#2ecc71" if pnl > 0 else "#e74c3c" for pnl in df["pnl"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(range(1, len(df) + 1)),
        y=df["pnl"],
        marker_color=colors,
        name="P&L",
    ))

    fig.update_layout(
        title="P&L by Trade",
        xaxis_title="Trade #",
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=350,
    )

    return fig


def main():
    # Load data
    data = load_trading_data()

    if not data["snapshots"]:
        st.warning("No paper trading data yet. Run the paper trading script first:")
        st.code("python scripts/paper_trade.py --symbol SPY")

        st.markdown("---")
        st.markdown("### Quick Start")
        st.markdown("""
        1. Make sure you have set up your API key in `.env`
        2. Run the paper trading script manually or set up GitHub Actions
        3. The script will fetch market data and execute paper trades
        4. Results will appear here
        """)
        return

    # Summary metrics
    st.markdown("## Portfolio Summary")

    if data["summary"]:
        summary = data["summary"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            equity = summary.get("equity", 0)
            st.metric(
                "Equity",
                f"${equity:,.0f}",
                f"{summary.get('cumulative_return', 0):.2%}",
            )

        with col2:
            st.metric(
                "Today's Return",
                f"{summary.get('daily_return', 0):.2%}",
            )

        with col3:
            st.metric(
                "Open Positions",
                summary.get("open_positions", 0),
            )

        with col4:
            vrp = summary.get("vrp", 0)
            st.metric(
                "Current VRP",
                f"{vrp:.2%}",
                "Bullish" if vrp > 0.02 else "Neutral",
            )

    # More detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Market Data")
        if data["summary"]:
            st.write(f"**IV:** {data['summary'].get('iv', 0):.2%}")
            st.write(f"**RV:** {data['summary'].get('rv', 0):.2%}")
            st.write(f"**VRP:** {data['summary'].get('vrp', 0):.2%}")
            st.write(f"**Last Update:** {data['summary'].get('date', 'N/A')}")

    with col2:
        st.markdown("### Performance Stats")
        if data["trades"]:
            trades_df = pd.DataFrame(data["trades"])
            wins = len(trades_df[trades_df["pnl"] > 0])
            total = len(trades_df)
            total_pnl = trades_df["pnl"].sum()

            st.write(f"**Total Trades:** {total}")
            st.write(f"**Win Rate:** {wins/total:.1%}" if total > 0 else "N/A")
            st.write(f"**Total P&L:** ${total_pnl:,.2f}")
            st.write(f"**Avg P&L:** ${total_pnl/total:,.2f}" if total > 0 else "N/A")

    st.markdown("---")

    # Charts
    st.markdown("## Performance Charts")

    # Equity curve
    equity_fig = plot_equity_curve(data["snapshots"])
    if equity_fig:
        st.plotly_chart(equity_fig, use_container_width=True)

    # Returns and P&L charts
    col1, col2 = st.columns(2)

    with col1:
        returns_fig = plot_returns_distribution(data["snapshots"])
        if returns_fig:
            st.plotly_chart(returns_fig, use_container_width=True)

    with col2:
        pnl_fig = plot_trades_pnl(data["trades"])
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True)

    st.markdown("---")

    # Open Positions
    st.markdown("## Open Positions")

    if data["state"] and data["state"].get("positions"):
        open_positions = [p for p in data["state"]["positions"] if p["status"] == "OPEN"]
        if open_positions:
            pos_df = pd.DataFrame(open_positions)
            pos_df = pos_df[["position_id", "entry_date", "position_type", "entry_iv",
                            "entry_rv", "entry_vrp", "target_exit_date", "notional_value"]]
            pos_df.columns = ["ID", "Entry Date", "Type", "Entry IV", "Entry RV",
                             "Entry VRP", "Target Exit", "Notional"]

            # Format
            pos_df["Entry IV"] = pos_df["Entry IV"].apply(lambda x: f"{x:.2%}")
            pos_df["Entry RV"] = pos_df["Entry RV"].apply(lambda x: f"{x:.2%}")
            pos_df["Entry VRP"] = pos_df["Entry VRP"].apply(lambda x: f"{x:.2%}")
            pos_df["Notional"] = pos_df["Notional"].apply(lambda x: f"${x:,.0f}")

            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")
    else:
        st.info("No open positions")

    st.markdown("---")

    # Trade History
    st.markdown("## Trade History")

    if data["trades"]:
        trades_df = pd.DataFrame(data["trades"])
        trades_df = trades_df.sort_values("exit_date", ascending=False)

        # Format display columns
        display_df = trades_df[["trade_id", "entry_date", "exit_date", "position_type",
                                "entry_iv", "exit_iv", "pnl", "pnl_pct", "holding_days"]].copy()
        display_df.columns = ["ID", "Entry", "Exit", "Type", "Entry IV", "Exit IV",
                             "P&L", "P&L %", "Days"]

        display_df["Entry IV"] = display_df["Entry IV"].apply(lambda x: f"{x:.2%}")
        display_df["Exit IV"] = display_df["Exit IV"].apply(lambda x: f"{x:.2%}")
        display_df["P&L"] = display_df["P&L"].apply(lambda x: f"${x:,.2f}")
        display_df["P&L %"] = display_df["P&L %"].apply(lambda x: f"{x:.2%}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades yet")

    # Daily Snapshots
    with st.expander("📊 Daily Snapshots"):
        if data["snapshots"]:
            snap_df = pd.DataFrame(data["snapshots"])
            snap_df = snap_df.sort_values("date", ascending=False)
            snap_df["equity"] = snap_df["equity"].apply(lambda x: f"${x:,.0f}")
            snap_df["daily_return"] = snap_df["daily_return"].apply(lambda x: f"{x:.2%}")
            snap_df["cumulative_return"] = snap_df["cumulative_return"].apply(lambda x: f"{x:.2%}")
            st.dataframe(snap_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
