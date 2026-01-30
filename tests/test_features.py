"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.volatility import VolatilityCalculator
from src.features.technical import TechnicalIndicators
from src.features.macro import MacroFeatures
from src.features.pipeline import FeaturePipeline


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator instance."""
        return VolatilityCalculator()

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 252

        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        returns = np.random.normal(0.0005, 0.015, n)
        prices = 100 * np.cumprod(1 + returns)

        return pd.Series(prices, index=dates)

    @pytest.fixture
    def sample_ohlc(self, sample_prices):
        """Create sample OHLC data."""
        close = sample_prices
        high = close * (1 + np.abs(np.random.normal(0, 0.01, len(close))))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, len(close))))
        open_ = close.shift(1).fillna(close.iloc[0])

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        })

    def test_realized_volatility(self, calculator, sample_prices):
        """Test realized volatility calculation."""
        rv = calculator.realized_volatility(sample_prices, window=21)

        assert len(rv) == len(sample_prices)
        assert rv.isna().sum() < 21  # Only first window should be NaN
        assert rv.dropna().min() > 0  # Volatility should be positive

    def test_parkinson_volatility(self, calculator, sample_ohlc):
        """Test Parkinson volatility calculation."""
        rv = calculator.realized_volatility_parkinson(
            sample_ohlc["high"],
            sample_ohlc["low"],
            window=21,
        )

        assert len(rv) == len(sample_ohlc)
        assert rv.dropna().min() > 0

    def test_garman_klass_volatility(self, calculator, sample_ohlc):
        """Test Garman-Klass volatility calculation."""
        rv = calculator.realized_volatility_garman_klass(
            sample_ohlc["open"],
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            window=21,
        )

        assert len(rv) == len(sample_ohlc)
        assert rv.dropna().min() > 0

    def test_yang_zhang_volatility(self, calculator, sample_ohlc):
        """Test Yang-Zhang volatility calculation."""
        rv = calculator.realized_volatility_yang_zhang(
            sample_ohlc["open"],
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            window=21,
        )

        assert len(rv) == len(sample_ohlc)

    def test_black_scholes_price(self, calculator):
        """Test Black-Scholes pricing."""
        # Call option
        call_price = calculator.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call"
        )
        assert call_price > 0

        # Put option
        put_price = calculator.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put"
        )
        assert put_price > 0

        # Put-call parity check
        pv_strike = 100 * np.exp(-0.05 * 0.25)
        assert abs((call_price - put_price) - (100 - pv_strike)) < 0.01

    def test_implied_volatility(self, calculator):
        """Test implied volatility calculation."""
        # Calculate a price first
        true_vol = 0.25
        price = calculator.black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=true_vol, option_type="call"
        )

        # Recover IV
        iv = calculator.implied_volatility(
            option_price=price, S=100, K=100, T=0.25, r=0.05, option_type="call"
        )

        assert iv is not None
        assert abs(iv - true_vol) < 0.001

    def test_volatility_risk_premium(self, calculator):
        """Test VRP calculation."""
        iv = pd.Series([0.20, 0.22, 0.25, 0.21])
        rv = pd.Series([0.18, 0.19, 0.22, 0.20])

        vrp = calculator.volatility_risk_premium(iv, rv)

        assert len(vrp) == 4
        assert (vrp == iv - rv).all()


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    @pytest.fixture
    def tech(self):
        """Create an instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100

        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.normal(0, 1, n))

        return pd.DataFrame({
            "open": close - np.random.uniform(0, 1, n),
            "high": close + np.random.uniform(0, 2, n),
            "low": close - np.random.uniform(0, 2, n),
            "close": close,
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=dates)

    def test_sma(self, tech, sample_data):
        """Test SMA calculation."""
        sma = tech.sma(sample_data["close"], window=20)

        assert len(sma) == len(sample_data)
        assert sma.iloc[19:].notna().all()  # Should have values after window

    def test_ema(self, tech, sample_data):
        """Test EMA calculation."""
        ema = tech.ema(sample_data["close"], span=20)

        assert len(ema) == len(sample_data)
        assert ema.notna().all()  # EMA starts from first value

    def test_rsi(self, tech, sample_data):
        """Test RSI calculation."""
        rsi = tech.rsi(sample_data["close"], window=14)

        assert len(rsi) == len(sample_data)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_atr(self, tech, sample_data):
        """Test ATR calculation."""
        atr = tech.atr(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
            window=14,
        )

        assert len(atr) == len(sample_data)
        assert atr.dropna().min() >= 0

    def test_bollinger_bands(self, tech, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = tech.bollinger_bands(sample_data["close"], window=20)

        assert len(upper) == len(sample_data)
        valid_idx = middle.dropna().index
        assert (upper.loc[valid_idx] >= middle.loc[valid_idx]).all()
        assert (middle.loc[valid_idx] >= lower.loc[valid_idx]).all()

    def test_macd(self, tech, sample_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = tech.macd(sample_data["close"])

        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

    def test_compute_all(self, tech, sample_data):
        """Test computing all indicators."""
        result = tech.compute_all(sample_data)

        # Should have many more columns
        assert len(result.columns) > len(sample_data.columns)
        # Should include expected indicators
        assert any("rsi" in col for col in result.columns)
        assert any("atr" in col for col in result.columns)


class TestMacroFeatures:
    """Tests for MacroFeatures class."""

    @pytest.fixture
    def macro(self):
        """Create an instance."""
        return MacroFeatures()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 252

        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.normal(0, 1, n))

        return pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=dates)

    def test_market_regime(self, macro, sample_data):
        """Test market regime detection."""
        regime = macro.market_regime(sample_data["close"])

        assert len(regime) == len(sample_data)
        assert set(regime.dropna().unique()).issubset({-1, 0, 1})

    def test_day_of_week(self, macro, sample_data):
        """Test day of week feature."""
        dow = macro.day_of_week(sample_data.index)

        assert len(dow) == len(sample_data)
        assert dow.min() >= 0
        assert dow.max() <= 4  # Weekdays only (B frequency)


class TestFeaturePipeline:
    """Tests for FeaturePipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance."""
        return FeaturePipeline()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 252

        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        returns = np.random.normal(0.0005, 0.015, n)
        close = 100 * np.cumprod(1 + returns)

        return pd.DataFrame({
            "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
            "close": close,
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=dates)

    def test_transform(self, pipeline, sample_data):
        """Test feature transformation."""
        features = pipeline.transform(sample_data)

        assert len(features) == len(sample_data)
        assert len(features.columns) > 10  # Should have many features

    def test_prepare_training_data(self, pipeline, sample_data):
        """Test preparing training data."""
        features = pipeline.transform(sample_data)
        features["rv_cc_21d"] = features.get(
            "rv_cc_21d",
            pd.Series(np.random.uniform(0.1, 0.3, len(features)), index=features.index),
        )

        X, y = pipeline.prepare_training_data(features, target_col="rv_cc_21d")

        # Should have fewer samples due to NaN dropping
        assert len(X) <= len(features)
        assert len(X) == len(y)
        # No NaN values
        assert not X.isna().any().any()
        assert not y.isna().any()
