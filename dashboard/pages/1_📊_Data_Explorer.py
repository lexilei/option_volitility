"""Data Explorer page for the dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")
st.markdown("Explore price data, options chains, and volatility metrics.")


@st.cache_data(ttl=3600)
def load_sample_data():
    """Load sample data for demonstration."""
    import numpy as np

    # Generate sample price data
    dates = pd.date_range(end=date.today(), periods=252 * 2, freq="B")
    np.random.seed(42)

    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        "close": prices,
        "volume": np.random.uniform(1e6, 5e6, len(dates)),
    })
    df = df.set_index("date")
    return df


def plot_candlestick(df: pd.DataFrame, title: str = "Price Chart"):
    """Create a candlestick chart."""
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
        height=500,
    )

    return fig


def plot_volume(df: pd.DataFrame):
    """Create a volume chart."""
    fig = px.bar(df.reset_index(), x="date", y="volume", title="Volume")
    fig.update_layout(template="plotly_white", height=300)
    return fig


def main():
    # Sidebar controls
    with st.sidebar:
        st.header("Data Settings")

        symbol = st.text_input("Symbol", value="SPY")
        days = st.slider("Days of History", 30, 730, 365)

        st.markdown("---")

        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Load from Cache", "Fetch from API"],
        )

        if data_source == "Fetch from API":
            api_key = st.text_input("Polygon API Key", type="password")
            fetch_button = st.button("Fetch Data")
        else:
            fetch_button = False

    # Main content
    tab1, tab2, tab3 = st.tabs(["Price Data", "Statistics", "Options Chain"])

    with tab1:
        st.subheader(f"{symbol} Price Data")

        # Load data
        if data_source == "Sample Data":
            df = load_sample_data()
            st.info("Showing sample data for demonstration purposes.")
        else:
            df = load_sample_data()  # Fallback to sample

        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=df.index.min().date(),
                min_value=df.index.min().date(),
                max_value=df.index.max().date(),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=df.index.max().date(),
                min_value=df.index.min().date(),
                max_value=df.index.max().date(),
            )

        # Filter data
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        filtered_df = df[mask]

        # Charts
        st.plotly_chart(plot_candlestick(filtered_df, f"{symbol} Price"), use_container_width=True)
        st.plotly_chart(plot_volume(filtered_df), use_container_width=True)

        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(filtered_df.tail(50), use_container_width=True)

    with tab2:
        st.subheader("Price Statistics")

        if "df" in dir() and not df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${filtered_df['close'].iloc[-1]:.2f}")
                st.metric(
                    "Change",
                    f"{(filtered_df['close'].iloc[-1] / filtered_df['close'].iloc[-2] - 1) * 100:.2f}%",
                )

            with col2:
                st.metric("52W High", f"${filtered_df['high'].max():.2f}")
                st.metric("52W Low", f"${filtered_df['low'].min():.2f}")

            with col3:
                returns = filtered_df["close"].pct_change().dropna()
                st.metric("Avg Daily Return", f"{returns.mean() * 100:.3f}%")
                st.metric("Daily Volatility", f"{returns.std() * 100:.2f}%")

            with col4:
                st.metric("Avg Volume", f"{filtered_df['volume'].mean() / 1e6:.1f}M")
                st.metric("Total Days", f"{len(filtered_df)}")

            # Returns distribution
            st.markdown("### Returns Distribution")
            returns_df = pd.DataFrame({"returns": returns * 100})
            fig = px.histogram(
                returns_df,
                x="returns",
                nbins=50,
                title="Daily Returns Distribution (%)",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Rolling statistics
            st.markdown("### Rolling Statistics")
            window = st.slider("Rolling Window (days)", 5, 63, 21)

            rolling_vol = returns.rolling(window).std() * (252 ** 0.5) * 100
            rolling_mean = filtered_df["close"].rolling(window).mean()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Annualized Volatility (%)")
            )
            fig.update_layout(
                title=f"{window}-Day Rolling Volatility",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Options Chain")
        st.info("Options chain data will be displayed when fetched from Polygon.io API.")

        # Sample options chain display
        st.markdown("### Sample Options Chain Structure")

        sample_options = pd.DataFrame({
            "Strike": [95, 100, 105, 110, 115],
            "Call Bid": [5.50, 2.80, 1.20, 0.45, 0.15],
            "Call Ask": [5.70, 3.00, 1.35, 0.55, 0.20],
            "Call IV": [0.22, 0.20, 0.19, 0.21, 0.24],
            "Put Bid": [0.20, 0.60, 1.50, 3.20, 5.80],
            "Put Ask": [0.25, 0.75, 1.65, 3.40, 6.00],
            "Put IV": [0.24, 0.21, 0.20, 0.19, 0.22],
        })

        st.dataframe(sample_options, use_container_width=True)

        # IV smile visualization
        st.markdown("### Volatility Smile")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_options["Strike"],
            y=sample_options["Call IV"],
            mode="lines+markers",
            name="Call IV",
        ))
        fig.add_trace(go.Scatter(
            x=sample_options["Strike"],
            y=sample_options["Put IV"],
            mode="lines+markers",
            name="Put IV",
        ))
        fig.update_layout(
            title="Implied Volatility Smile",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
