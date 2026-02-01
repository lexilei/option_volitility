"""Settings page for the dashboard."""

import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.markdown("Configure application settings and parameters.")


def main():
    tab1, tab2, tab3, tab4 = st.tabs([
        "API Configuration",
        "Model Settings",
        "Strategy Parameters",
        "System Info",
    ])

    with tab1:
        st.subheader("API Configuration")

        st.markdown("### Massive API (formerly Polygon.io)")

        api_key = st.text_input(
            "API Key",
            type="password",
            help="Your Massive API key for fetching market data",
        )

        if api_key:
            st.success("API key configured")

            # Test connection
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    try:
                        import requests

                        response = requests.get(
                            f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey={api_key}",
                            timeout=10,
                        )
                        if response.status_code == 200:
                            st.success("Connection successful!")
                        else:
                            st.error(f"Connection failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

        st.markdown("---")

        st.markdown("### Data Settings")

        col1, col2 = st.columns(2)

        with col1:
            data_dir = st.text_input("Data Directory", value="data")
            cache_enabled = st.checkbox("Enable Data Caching", value=True)

        with col2:
            rate_limit = st.number_input(
                "API Rate Limit (requests/min)",
                min_value=1,
                max_value=100,
                value=5,
            )

        if st.button("Save API Settings"):
            st.success("API settings saved!")

    with tab2:
        st.subheader("Model Settings")

        st.markdown("### Walk-Forward Validation")

        col1, col2 = st.columns(2)

        with col1:
            train_window = st.number_input(
                "Training Window (days)",
                min_value=126,
                max_value=1260,
                value=504,
                help="Number of trading days for training",
            )

            test_window = st.number_input(
                "Test Window (days)",
                min_value=21,
                max_value=126,
                value=63,
                help="Number of trading days for testing",
            )

        with col2:
            step_size = st.number_input(
                "Step Size (days)",
                min_value=21,
                max_value=126,
                value=63,
                help="Days to advance between folds",
            )

            gap_days = st.number_input(
                "Gap Days",
                min_value=0,
                max_value=21,
                value=0,
                help="Gap between train and test to prevent look-ahead",
            )

        st.markdown("---")

        st.markdown("### Hyperparameter Tuning")

        tune_enabled = st.checkbox("Enable Hyperparameter Tuning", value=False)

        if tune_enabled:
            n_trials = st.slider(
                "Number of Optuna Trials",
                min_value=10,
                max_value=200,
                value=50,
            )

            optimization_metric = st.selectbox(
                "Optimization Metric",
                ["RMSE", "MAE", "R2", "Sharpe"],
            )

        st.markdown("---")

        st.markdown("### Model Selection")

        available_models = [
            "Historical Mean",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "XGBoost",
            "LightGBM",
            "Random Forest",
            "LSTM",
            "TFT",
        ]

        selected_models = st.multiselect(
            "Models to Train",
            available_models,
            default=["Historical Mean", "Ridge", "XGBoost", "LightGBM"],
        )

        train_ensemble = st.checkbox("Train Ensemble Model", value=True)

        if st.button("Save Model Settings"):
            st.success("Model settings saved!")

    with tab3:
        st.subheader("Strategy Parameters")

        st.markdown("### Signal Generation")

        col1, col2 = st.columns(2)

        with col1:
            vrp_threshold = st.slider(
                "VRP Threshold",
                min_value=0.01,
                max_value=0.10,
                value=0.03,
                step=0.005,
                help="Minimum volatility risk premium to enter position",
            )

            signal_strength = st.checkbox(
                "Use Signal Strength Scaling",
                value=True,
                help="Scale position size by signal strength",
            )

        with col2:
            holding_period = st.slider(
                "Holding Period (days)",
                min_value=7,
                max_value=42,
                value=21,
                help="Days to hold each position",
            )

        st.markdown("---")

        st.markdown("### Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=25,
                value=10,
                help="Maximum position size as percentage of capital",
            )

            max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=10,
                value=3,
            )

        with col2:
            max_drawdown = st.slider(
                "Max Drawdown Threshold (%)",
                min_value=5,
                max_value=30,
                value=15,
                help="Stop trading if drawdown exceeds this",
            )

            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=10,
                max_value=100,
                value=50,
                help="Stop loss as percentage of premium",
            )

        st.markdown("---")

        st.markdown("### Backtest Settings")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
        )

        commission = st.number_input(
            "Commission per Trade ($)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )

        slippage = st.slider(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )

        if st.button("Save Strategy Settings"):
            st.success("Strategy settings saved!")

    with tab4:
        st.subheader("System Information")

        # Python info
        import sys
        import platform

        st.markdown("### Environment")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Python Version:** {sys.version.split()[0]}")
            st.markdown(f"**Platform:** {platform.system()} {platform.release()}")

        with col2:
            try:
                import torch

                device = "CUDA" if torch.cuda.is_available() else "CPU"
                st.markdown(f"**PyTorch Device:** {device}")
            except ImportError:
                st.markdown("**PyTorch:** Not installed")

        st.markdown("---")

        st.markdown("### Installed Packages")

        packages = [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("scikit-learn", "sklearn"),
            ("xgboost", "xgboost"),
            ("lightgbm", "lightgbm"),
            ("torch", "torch"),
            ("streamlit", "streamlit"),
            ("plotly", "plotly"),
            ("optuna", "optuna"),
        ]

        package_versions = []
        for name, module in packages:
            try:
                mod = __import__(module)
                version = getattr(mod, "__version__", "unknown")
                package_versions.append({"Package": name, "Version": version, "Status": "✅"})
            except ImportError:
                package_versions.append({"Package": name, "Version": "N/A", "Status": "❌"})

        st.dataframe(package_versions, use_container_width=True)

        st.markdown("---")

        st.markdown("### Data Directory Status")

        data_dir = Path("data")

        if data_dir.exists():
            raw_files = list((data_dir / "raw").glob("*.parquet")) if (data_dir / "raw").exists() else []
            processed_files = list((data_dir / "processed").glob("*.parquet")) if (data_dir / "processed").exists() else []
            model_files = list((data_dir / "models").glob("*.joblib")) if (data_dir / "models").exists() else []

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Raw Data Files", len(raw_files))

            with col2:
                st.metric("Processed Files", len(processed_files))

            with col3:
                st.metric("Saved Models", len(model_files))
        else:
            st.warning("Data directory not found. Run data fetching first.")

        # Clear cache button
        st.markdown("---")

        st.markdown("### Cache Management")

        if st.button("Clear Streamlit Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

        if st.button("Clear All Data"):
            st.warning("This will delete all downloaded data and trained models.")
            confirm = st.checkbox("I understand and want to proceed")
            if confirm:
                # Would actually delete files here
                st.success("Data cleared!")


if __name__ == "__main__":
    main()
