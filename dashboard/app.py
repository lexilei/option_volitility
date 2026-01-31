"""Main Streamlit dashboard application."""

import streamlit as st
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Vol Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - works with both light and dark themes
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    /* Metric styling that works with dark theme */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
    }
    [data-testid="stMetricLabel"] {
        color: #888 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FAFAFA !important;
    }
    </style>
""", unsafe_allow_html=True)


def load_paper_trading_summary():
    """Load paper trading summary if available."""
    summary_file = Path("data/paper_trading/latest_summary.json")
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def main():
    """Main application entry point."""
    st.markdown('<p class="main-header">📈 Vol Strategy Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">ML-Enhanced Volatility Selling Strategy with Live Paper Trading</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        st.markdown("""
        - 📊 **Data Explorer** - View market data
        - 🔬 **Feature Analysis** - Explore features
        - 🤖 **Model Comparison** - Compare ML models
        - 📈 **Backtest Results** - Historical backtests
        - 💰 **Paper Trading** - Live simulation
        - ⚙️ **Settings** - Configuration
        """)

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

        # Load actual paper trading data
        summary = load_paper_trading_summary()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "7")
        with col2:
            st.metric("Features", "90+")

        if summary:
            st.markdown("---")
            st.markdown("### 💰 Paper Trading")
            col1, col2 = st.columns(2)
            with col1:
                ret = summary.get("cumulative_return", 0)
                st.metric("Return", f"{ret:.1%}")
            with col2:
                positions = summary.get("open_positions", 0)
                st.metric("Positions", positions)

    # Main content
    st.markdown("## 👋 Welcome")

    st.markdown("""
    This dashboard provides tools for analyzing and comparing volatility prediction models
    for options trading strategies. The core strategy exploits the **Volatility Risk Premium (VRP)**
    — the tendency for implied volatility to exceed realized volatility.
    """)

    # Paper Trading Status Card
    summary = load_paper_trading_summary()
    if summary:
        st.markdown("---")
        st.markdown("## 💰 Live Paper Trading Status")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            equity = summary.get("equity", 100000)
            st.metric("Portfolio Value", f"${equity:,.0f}")
        with col2:
            ret = summary.get("cumulative_return", 0)
            st.metric("Total Return", f"{ret:.2%}")
        with col3:
            vrp = summary.get("vrp", 0)
            st.metric("Current VRP", f"{vrp:.2%}")
        with col4:
            positions = summary.get("open_positions", 0)
            st.metric("Open Positions", positions)

        st.caption(f"Last updated: {summary.get('date', 'N/A')}")

    st.markdown("---")

    # Key Features
    st.markdown("## ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Data & Analysis
        - **Real-time data** from Massive (Polygon.io) API
        - **90+ features** including volatility, technical, and macro indicators
        - **Multiple RV estimators** (Close-to-close, Parkinson, Yang-Zhang)
        - **IV calculation** from options chain
        """)

        st.markdown("""
        ### 🤖 ML Models
        - Historical Mean (baseline)
        - Ridge/Lasso regression
        - XGBoost/LightGBM
        - LSTM neural network
        - Temporal Fusion Transformer
        - Ensemble methods
        """)

    with col2:
        st.markdown("""
        ### 📈 Backtesting
        - **Walk-forward validation** to avoid look-ahead bias
        - **Performance metrics**: Sharpe, Sortino, Max Drawdown
        - **Risk management** with position sizing
        - **Detailed trade analysis**
        """)

        st.markdown("""
        ### 💰 Paper Trading
        - **Live simulation** with real market data
        - **Automated daily updates** via GitHub Actions
        - **Position tracking** and P&L monitoring
        - **Performance dashboard** with equity curve
        """)

    st.markdown("---")

    # Getting Started
    st.markdown("## 🚀 Getting Started")

    st.markdown("""
    1. **Explore Data** → Go to **Data Explorer** to view price data and volatility metrics
    2. **Analyze Features** → Check **Feature Analysis** for correlation and importance
    3. **Compare Models** → Use **Model Comparison** to evaluate ML models
    4. **Review Backtests** → See historical performance in **Backtest Results**
    5. **Monitor Live Trading** → Track paper trading in **Paper Trading** page
    6. **Configure** → Adjust settings in **Settings** page
    """)

    st.markdown("---")

    # Overview cards
    st.markdown("## 📋 Strategy Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Signal Generation")
        st.markdown("""
        - Calculate VRP = IV - Predicted RV
        - **SELL VOL** when VRP > threshold
        - **BUY VOL** when VRP < -threshold
        - Default threshold: 2%
        """)

    with col2:
        st.markdown("### ⚖️ Risk Management")
        st.markdown("""
        - Max 5 concurrent positions
        - $10,000 per position
        - 21-day holding period
        - Stop-loss protection
        """)

    with col3:
        st.markdown("### 📊 Performance Tracking")
        st.markdown("""
        - Daily equity snapshots
        - Trade-by-trade analysis
        - Win rate & profit factor
        - Drawdown monitoring
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Vol Strategy Dashboard | "
        "<a href='https://github.com/lexilei/option_volitility' style='color: #FF6B6B;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
