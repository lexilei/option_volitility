"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_days = 252

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n_days),
    }, index=dates)

    return df


@pytest.fixture(scope="session")
def sample_features(sample_price_data):
    """Generate sample features for testing."""
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(
        rv_windows=[5, 21],
        include_technical=True,
        include_macro=True,
        include_lags=False,
    )

    return pipeline.transform(sample_price_data)


@pytest.fixture(scope="session")
def sample_training_data(sample_features):
    """Generate sample training data."""
    from src.features.volatility import VolatilityCalculator

    vol_calc = VolatilityCalculator()

    # Add target if not present
    if "rv_cc_21d" not in sample_features.columns:
        sample_features = sample_features.copy()
        sample_features["rv_cc_21d"] = vol_calc.realized_volatility(
            pd.Series(100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(sample_features))),
                     index=sample_features.index),
            window=21
        )

    # Prepare X and y
    target_col = "rv_cc_21d"
    y = sample_features[target_col].shift(-21)

    feature_cols = [c for c in sample_features.columns if c != target_col]
    X = sample_features[feature_cols]

    # Drop NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_key():
    """Return a mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture(scope="session")
def sample_options_data():
    """Generate sample options chain data."""
    np.random.seed(42)

    strikes = np.arange(95, 110, 1)
    expirations = pd.date_range(start="2024-01-19", periods=4, freq="ME")

    records = []
    for exp in expirations:
        for strike in strikes:
            for opt_type in ["call", "put"]:
                records.append({
                    "ticker": f"O:SPY{exp.strftime('%y%m%d')}{opt_type[0].upper()}{strike:08d}",
                    "underlying_ticker": "SPY",
                    "contract_type": opt_type,
                    "strike_price": strike,
                    "expiration_date": exp.strftime("%Y-%m-%d"),
                    "implied_volatility": np.random.uniform(0.15, 0.35),
                    "bid": np.random.uniform(0.5, 5.0),
                    "ask": np.random.uniform(0.6, 5.5),
                    "volume": np.random.randint(100, 10000),
                    "open_interest": np.random.randint(1000, 50000),
                })

    return pd.DataFrame(records)


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve for testing."""
    np.random.seed(42)
    n_days = 252

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0008, 0.012, n_days)
    equity = 100000 * np.cumprod(1 + returns)

    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_trades():
    """Generate sample trades for testing."""
    np.random.seed(42)
    n_trades = 20

    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    entry_dates = np.random.choice(dates[:-30], n_trades, replace=False)
    entry_dates = sorted(entry_dates)

    trades = []
    for i, entry_date in enumerate(entry_dates):
        exit_date = entry_date + pd.Timedelta(days=np.random.randint(14, 28))
        pnl = np.random.normal(500, 2000)

        trades.append({
            "trade_id": i + 1,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "signal": np.random.choice(["SELL_VOL", "BUY_VOL"]),
            "pnl": pnl,
            "return_pct": pnl / 10000,
        })

    return pd.DataFrame(trades)
