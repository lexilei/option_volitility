"""Feature Analysis page for the dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Feature Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Feature Analysis")
st.markdown("Analyze computed features and their relationships.")


@st.cache_data(ttl=3600)
def generate_sample_features():
    """Generate sample feature data for demonstration."""
    np.random.seed(42)
    n_samples = 504  # 2 years of trading days

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_samples, freq="B")

    # Base price series
    returns = np.random.normal(0.0005, 0.015, n_samples)
    prices = 100 * np.cumprod(1 + returns)

    # Realized volatility (various windows)
    rv_5d = pd.Series(returns).rolling(5).std() * np.sqrt(252)
    rv_21d = pd.Series(returns).rolling(21).std() * np.sqrt(252)
    rv_63d = pd.Series(returns).rolling(63).std() * np.sqrt(252)

    # Implied volatility (with premium)
    iv = rv_21d + np.random.normal(0.02, 0.01, n_samples)
    iv = np.clip(iv, 0.05, 0.8)

    # VRP
    vrp = iv - rv_21d

    # Technical indicators
    rsi = 50 + np.cumsum(np.random.normal(0, 5, n_samples))
    rsi = np.clip(rsi, 0, 100)

    atr = prices * (0.01 + np.abs(np.random.normal(0, 0.005, n_samples)))

    # VIX proxy
    vix = 15 + np.cumsum(np.random.normal(0, 1, n_samples))
    vix = np.clip(vix, 10, 80)

    df = pd.DataFrame({
        "date": dates,
        "close": prices,
        "return_1d": returns,
        "rv_5d": rv_5d,
        "rv_21d": rv_21d,
        "rv_63d": rv_63d,
        "iv": iv,
        "vrp": vrp,
        "rsi_14": rsi,
        "atr_14": atr,
        "vix": vix,
        "volume_ratio": np.random.lognormal(0, 0.3, n_samples),
        "momentum_21d": pd.Series(prices).pct_change(21).values,
    })

    df = df.set_index("date")
    return df


def plot_feature_time_series(df: pd.DataFrame, columns: list[str], title: str):
    """Plot multiple features over time."""
    fig = make_subplots(
        rows=len(columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=columns,
    )

    for i, col in enumerate(columns, 1):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col, mode="lines"),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=200 * len(columns),
        title=title,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str]):
    """Plot correlation heatmap."""
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
        title="Feature Correlation Matrix",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_feature_distribution(df: pd.DataFrame, column: str):
    """Plot feature distribution."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Distribution", "Box Plot"],
    )

    fig.add_trace(
        go.Histogram(x=df[column].dropna(), nbinsx=50, name="Distribution"),
        row=1,
        col=1,
    )

    fig.add_trace(go.Box(y=df[column].dropna(), name=column), row=1, col=2)

    fig.update_layout(
        title=f"{column} Distribution",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def main():
    # Sidebar
    with st.sidebar:
        st.header("Feature Settings")

        feature_category = st.selectbox(
            "Feature Category",
            ["Volatility", "Technical", "Macro", "All"],
        )

        st.markdown("---")

        show_correlations = st.checkbox("Show Correlations", value=True)
        show_distributions = st.checkbox("Show Distributions", value=True)

    # Load data
    df = generate_sample_features()

    # Feature groups
    vol_features = ["rv_5d", "rv_21d", "rv_63d", "iv", "vrp"]
    tech_features = ["rsi_14", "atr_14", "volume_ratio", "momentum_21d"]
    macro_features = ["vix"]

    if feature_category == "Volatility":
        selected_features = vol_features
    elif feature_category == "Technical":
        selected_features = tech_features
    elif feature_category == "Macro":
        selected_features = macro_features
    else:
        selected_features = vol_features + tech_features + macro_features

    # Main content
    tab1, tab2, tab3 = st.tabs(["Time Series", "Correlations", "Distributions"])

    with tab1:
        st.subheader("Feature Time Series")

        # Feature selector
        features_to_plot = st.multiselect(
            "Select Features to Plot",
            selected_features,
            default=selected_features[:3],
        )

        if features_to_plot:
            st.plotly_chart(
                plot_feature_time_series(df, features_to_plot, "Feature Evolution"),
                use_container_width=True,
            )

        # Summary statistics
        st.markdown("### Summary Statistics")
        stats_df = df[selected_features].describe().T
        stats_df["skew"] = df[selected_features].skew()
        stats_df["kurtosis"] = df[selected_features].kurtosis()
        st.dataframe(stats_df.round(4), use_container_width=True)

    with tab2:
        st.subheader("Feature Correlations")

        if show_correlations:
            st.plotly_chart(
                plot_correlation_matrix(df, selected_features),
                use_container_width=True,
            )

            # Top correlations table
            st.markdown("### Strongest Correlations")
            corr = df[selected_features].corr()
            corr_pairs = []

            for i, col1 in enumerate(selected_features):
                for j, col2 in enumerate(selected_features):
                    if i < j:
                        corr_pairs.append({
                            "Feature 1": col1,
                            "Feature 2": col2,
                            "Correlation": corr.loc[col1, col2],
                        })

            corr_df = pd.DataFrame(corr_pairs)
            corr_df["Abs Correlation"] = corr_df["Correlation"].abs()
            corr_df = corr_df.sort_values("Abs Correlation", ascending=False)
            st.dataframe(corr_df.head(10).round(4), use_container_width=True)

    with tab3:
        st.subheader("Feature Distributions")

        if show_distributions:
            # Select feature for detailed analysis
            selected_feature = st.selectbox(
                "Select Feature for Detailed Analysis",
                selected_features,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    plot_feature_distribution(df, selected_feature),
                    use_container_width=True,
                )

            with col2:
                # Statistics
                feature_data = df[selected_feature].dropna()

                st.markdown(f"### {selected_feature} Statistics")
                st.metric("Mean", f"{feature_data.mean():.4f}")
                st.metric("Median", f"{feature_data.median():.4f}")
                st.metric("Std Dev", f"{feature_data.std():.4f}")
                st.metric("Skewness", f"{feature_data.skew():.4f}")
                st.metric("Kurtosis", f"{feature_data.kurtosis():.4f}")

            # Feature vs Target scatter
            st.markdown("### Feature vs Future Volatility")

            # Create forward RV as target
            target = df["rv_21d"].shift(-21)
            valid_mask = ~(df[selected_feature].isna() | target.isna())

            scatter_df = pd.DataFrame({
                selected_feature: df.loc[valid_mask, selected_feature],
                "Future RV (21d)": target[valid_mask],
            })

            fig = px.scatter(
                scatter_df,
                x=selected_feature,
                y="Future RV (21d)",
                opacity=0.5,
                trendline="ols",
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
