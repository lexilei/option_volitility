"""Model Comparison page for the dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")

st.title("🤖 Model Comparison")
st.markdown("Compare performance of different volatility prediction models.")


@st.cache_data(ttl=3600)
def generate_sample_model_results():
    """Generate sample model comparison data."""
    np.random.seed(42)

    models = [
        "Historical Mean",
        "Ridge",
        "Lasso",
        "XGBoost",
        "LightGBM",
        "LSTM",
        "TFT",
        "Ensemble",
    ]

    # Performance metrics
    metrics_data = {
        "Model": models,
        "RMSE": [0.045, 0.038, 0.039, 0.032, 0.031, 0.035, 0.030, 0.029],
        "MAE": [0.035, 0.029, 0.030, 0.025, 0.024, 0.028, 0.023, 0.022],
        "R2": [0.15, 0.32, 0.30, 0.45, 0.47, 0.40, 0.52, 0.55],
        "MAPE (%)": [18.5, 14.2, 14.8, 11.5, 11.0, 12.8, 10.2, 9.8],
        "Sharpe": [0.45, 0.82, 0.78, 1.15, 1.22, 0.95, 1.35, 1.42],
        "Max DD (%)": [15.2, 12.5, 13.0, 9.8, 9.2, 11.5, 8.5, 8.0],
        "Win Rate (%)": [52, 55, 54, 58, 59, 56, 61, 62],
    }
    metrics_df = pd.DataFrame(metrics_data).set_index("Model")

    # Walk-forward results by fold
    n_folds = 8
    fold_results = []
    for model in models:
        base_rmse = metrics_df.loc[model, "RMSE"]
        for fold in range(n_folds):
            fold_results.append({
                "Model": model,
                "Fold": fold + 1,
                "RMSE": base_rmse * (1 + np.random.uniform(-0.15, 0.15)),
                "R2": metrics_df.loc[model, "R2"] * (1 + np.random.uniform(-0.2, 0.2)),
            })
    fold_df = pd.DataFrame(fold_results)

    # Cumulative returns (simulated)
    n_days = 504
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    returns_data = {"date": dates}

    for model in models:
        sharpe = metrics_df.loc[model, "Sharpe"]
        daily_return = sharpe * 0.15 / np.sqrt(252)  # Approximate daily return
        returns = np.random.normal(daily_return, 0.01, n_days)
        cumulative = np.cumprod(1 + returns)
        returns_data[model] = cumulative

    returns_df = pd.DataFrame(returns_data).set_index("date")

    # Feature importance (for tree models)
    features = ["rv_21d", "iv", "vrp", "rsi_14", "vix", "atr_14", "momentum", "volume"]
    importance_data = []
    for model in ["XGBoost", "LightGBM"]:
        importance = np.random.dirichlet(np.ones(len(features)) * 2)
        for feat, imp in zip(features, importance):
            importance_data.append({
                "Model": model,
                "Feature": feat,
                "Importance": imp,
            })
    importance_df = pd.DataFrame(importance_data)

    return metrics_df, fold_df, returns_df, importance_df


def plot_metrics_comparison(metrics_df: pd.DataFrame, metric: str):
    """Create bar chart for metric comparison."""
    fig = px.bar(
        metrics_df.reset_index(),
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} by Model",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_cumulative_returns(returns_df: pd.DataFrame, models: list[str]):
    """Plot cumulative returns for selected models."""
    fig = go.Figure()

    for model in models:
        if model in returns_df.columns:
            fig.add_trace(go.Scatter(
                x=returns_df.index,
                y=returns_df[model],
                mode="lines",
                name=model,
            ))

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_fold_results(fold_df: pd.DataFrame, metric: str):
    """Plot fold-by-fold results."""
    fig = px.box(
        fold_df,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Distribution Across Folds",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame):
    """Plot feature importance comparison."""
    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        color="Model",
        barmode="group",
        title="Feature Importance Comparison",
    )
    fig.update_layout(template="plotly_white")
    return fig


def main():
    # Load data
    metrics_df, fold_df, returns_df, importance_df = generate_sample_model_results()

    # Sidebar
    with st.sidebar:
        st.header("Model Settings")

        selected_models = st.multiselect(
            "Select Models",
            list(metrics_df.index),
            default=list(metrics_df.index),
        )

        st.markdown("---")

        primary_metric = st.selectbox(
            "Primary Metric",
            ["RMSE", "MAE", "R2", "Sharpe", "Win Rate (%)"],
            index=3,
        )

        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown("""
        - **Historical Mean**: Baseline model
        - **Ridge/Lasso**: Linear with regularization
        - **XGBoost/LightGBM**: Gradient boosting
        - **LSTM**: Recurrent neural network
        - **TFT**: Transformer architecture
        - **Ensemble**: Weighted combination
        """)

    # Filter data
    filtered_metrics = metrics_df.loc[selected_models]
    filtered_returns = returns_df[selected_models]

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Summary",
        "Walk-Forward Results",
        "Equity Curves",
        "Feature Importance",
    ])

    with tab1:
        st.subheader("Model Performance Summary")

        # Highlight best model
        best_model = filtered_metrics[primary_metric].idxmax() if "R2" in primary_metric or "Sharpe" in primary_metric or "Win" in primary_metric else filtered_metrics[primary_metric].idxmin()
        st.success(f"Best model by {primary_metric}: **{best_model}**")

        # Metrics table
        st.dataframe(
            filtered_metrics.style.highlight_max(axis=0, subset=["R2", "Sharpe", "Win Rate (%)"]).highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE (%)", "Max DD (%)"]),
            use_container_width=True,
        )

        # Metric comparison charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_metrics_comparison(filtered_metrics, "RMSE"),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                plot_metrics_comparison(filtered_metrics, "Sharpe"),
                use_container_width=True,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                plot_metrics_comparison(filtered_metrics, "R2"),
                use_container_width=True,
            )

        with col4:
            st.plotly_chart(
                plot_metrics_comparison(filtered_metrics, "Win Rate (%)"),
                use_container_width=True,
            )

    with tab2:
        st.subheader("Walk-Forward Cross-Validation Results")

        st.markdown("""
        Walk-forward validation trains on historical data and tests on future periods,
        simulating real trading conditions. Each fold represents a test period.
        """)

        # Fold-by-fold comparison
        col1, col2 = st.columns(2)

        with col1:
            filtered_fold = fold_df[fold_df["Model"].isin(selected_models)]
            st.plotly_chart(
                plot_fold_results(filtered_fold, "RMSE"),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                plot_fold_results(filtered_fold, "R2"),
                use_container_width=True,
            )

        # Detailed fold results table
        st.markdown("### Detailed Fold Results")

        pivot_df = fold_df[fold_df["Model"].isin(selected_models)].pivot(
            index="Fold",
            columns="Model",
            values="RMSE",
        )
        st.dataframe(pivot_df.round(4), use_container_width=True)

    with tab3:
        st.subheader("Cumulative Returns")

        st.plotly_chart(
            plot_cumulative_returns(filtered_returns, selected_models),
            use_container_width=True,
        )

        # Drawdown analysis
        st.markdown("### Drawdown Analysis")

        # Calculate drawdowns
        drawdown_data = {}
        for model in selected_models:
            if model in returns_df.columns:
                equity = returns_df[model]
                peak = equity.expanding().max()
                drawdown = (peak - equity) / peak
                drawdown_data[model] = drawdown

        drawdown_df = pd.DataFrame(drawdown_data)

        fig = go.Figure()
        for model in selected_models:
            if model in drawdown_df.columns:
                fig.add_trace(go.Scatter(
                    x=drawdown_df.index,
                    y=drawdown_df[model] * 100,
                    fill="tozeroy",
                    name=model,
                    mode="lines",
                ))

        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Feature Importance")

        st.markdown("""
        Feature importance shows which inputs have the most predictive power.
        Only available for tree-based models (XGBoost, LightGBM).
        """)

        st.plotly_chart(
            plot_feature_importance(importance_df),
            use_container_width=True,
        )

        # Top features table
        st.markdown("### Top Features by Model")

        for model in ["XGBoost", "LightGBM"]:
            if model in selected_models:
                model_imp = importance_df[importance_df["Model"] == model].sort_values(
                    "Importance", ascending=False
                )
                st.markdown(f"**{model}**")
                st.dataframe(model_imp[["Feature", "Importance"]].round(4), use_container_width=True)


if __name__ == "__main__":
    main()
