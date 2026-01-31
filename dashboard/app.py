"""Main Streamlit dashboard application."""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Option Volatility Strategy",
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
        margin-bottom: 1rem;
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


def main():
    """Main application entry point."""
    st.markdown('<p class="main-header">Option Volatility Strategy</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">ML-Enhanced Volatility Selling Strategy Dashboard</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=VolStrat", width=150)
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("""
        Use the sidebar to navigate between pages:
        - **Data Explorer**: View and analyze raw data
        - **Feature Analysis**: Explore computed features
        - **Model Comparison**: Compare ML model performance
        - **Backtest Results**: View strategy backtests
        - **Settings**: Configure parameters
        """)

        st.markdown("---")
        st.markdown("### Quick Stats")

        # Placeholder stats - would be loaded from actual data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "7")
        with col2:
            st.metric("Features", "50+")

    # Main content
    st.markdown("## Welcome")

    st.markdown("""
    This dashboard provides tools for analyzing and comparing volatility prediction models
    for options trading strategies. The core strategy exploits the **Volatility Risk Premium (VRP)**
    - the tendency for implied volatility to exceed realized volatility.

    ### Key Features

    - **Data Exploration**: Visualize price data, options chains, and volatility metrics
    - **Feature Engineering**: Analyze computed features including technical indicators and macro factors
    - **Model Comparison**: Compare performance of multiple ML models (Ridge, Lasso, XGBoost, LSTM, TFT)
    - **Backtesting**: Evaluate strategy performance with walk-forward validation

    ### Getting Started

    1. Configure your API keys in the **Settings** page
    2. Fetch data using the **Data Explorer** page
    3. Review features in **Feature Analysis**
    4. Compare models in **Model Comparison**
    5. Analyze results in **Backtest Results**
    """)

    st.markdown("---")

    # Quick overview cards
    st.markdown("## Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Data Pipeline")
        st.markdown("""
        - Polygon.io API integration
        - OHLCV price data
        - Options chain data
        - VIX and macro data
        """)

    with col2:
        st.markdown("### ML Models")
        st.markdown("""
        - Historical Mean (baseline)
        - Ridge/Lasso regression
        - XGBoost/LightGBM
        - LSTM neural network
        - Temporal Fusion Transformer
        """)

    with col3:
        st.markdown("### Strategy")
        st.markdown("""
        - VRP-based signals
        - Walk-forward validation
        - Risk management
        - Performance metrics
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Option Volatility Strategy Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
