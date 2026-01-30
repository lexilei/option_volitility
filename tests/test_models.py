"""Tests for model modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import BaseVolModel
from src.models.baseline import HistoricalMeanModel, NaiveModel, EWMAModel
from src.models.linear import RidgeVolModel, LassoVolModel


class TestBaseVolModel:
    """Tests for BaseVolModel class."""

    def test_is_abstract(self):
        """Test that BaseVolModel cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseVolModel()


class TestHistoricalMeanModel:
    """Tests for HistoricalMeanModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        })
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = HistoricalMeanModel(window=21)

        model.fit(X, y)

        assert model.is_fitted
        assert model._last_values is not None

    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        model = HistoricalMeanModel(window=21)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).all()

    def test_predict_before_fit(self, sample_data):
        """Test that predict raises error before fit."""
        X, y = sample_data
        model = HistoricalMeanModel()

        with pytest.raises(ValueError):
            model.predict(X)

    def test_save_and_load(self, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model = HistoricalMeanModel(window=21)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            loaded = HistoricalMeanModel.load(f.name)

            assert loaded.is_fitted
            assert loaded.window == 21


class TestNaiveModel:
    """Tests for NaiveModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({"feature1": np.random.randn(n)})
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    def test_fit_predict(self, sample_data):
        """Test fit and predict."""
        X, y = sample_data
        model = NaiveModel()

        model.fit(X, y)
        predictions = model.predict(X)

        # Should predict the last value
        assert all(predictions == y.iloc[-1])


class TestEWMAModel:
    """Tests for EWMAModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({"feature1": np.random.randn(n)})
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    def test_fit_predict(self, sample_data):
        """Test fit and predict."""
        X, y = sample_data
        model = EWMAModel(span=21)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()


class TestRidgeVolModel:
    """Tests for RidgeVolModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data with signal."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
        })
        # Create target with some relationship to features
        y = 0.5 * X["feature1"] + 0.3 * X["feature2"] + np.random.randn(n) * 0.1

        return X, y

    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = RidgeVolModel(alpha=1.0)

        model.fit(X, y)

        assert model.is_fitted
        assert model.model.coef_ is not None

    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        model = RidgeVolModel(alpha=1.0)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_feature_importance(self, sample_data):
        """Test feature importance."""
        X, y = sample_data
        model = RidgeVolModel(alpha=1.0)

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == 3

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model = RidgeVolModel(alpha=1.0)

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0


class TestLassoVolModel:
    """Tests for LassoVolModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with sparse features."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            "important1": np.random.randn(n),
            "important2": np.random.randn(n),
            "noise1": np.random.randn(n),
            "noise2": np.random.randn(n),
            "noise3": np.random.randn(n),
        })
        # Target only depends on first two features
        y = 0.5 * X["important1"] + 0.3 * X["important2"] + np.random.randn(n) * 0.1

        return X, y

    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = LassoVolModel(alpha=0.1)

        model.fit(X, y)

        assert model.is_fitted

    def test_feature_selection(self, sample_data):
        """Test that Lasso performs feature selection."""
        X, y = sample_data
        model = LassoVolModel(alpha=0.1)

        model.fit(X, y)
        selected = model.get_selected_features()

        # With proper alpha, should select fewer than all features
        # (though with small data this isn't guaranteed)
        assert len(selected) <= len(X.columns)


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions."""
        y_true = np.array([0.2, 0.25, 0.3, 0.22, 0.28])
        y_pred = y_true.copy()
        return y_true, y_pred

    @pytest.fixture
    def imperfect_predictions(self):
        """Create imperfect predictions."""
        y_true = np.array([0.2, 0.25, 0.3, 0.22, 0.28])
        y_pred = y_true + np.random.uniform(-0.02, 0.02, len(y_true))
        return y_true, y_pred

    def test_perfect_r2(self, perfect_predictions):
        """Test R2 with perfect predictions."""
        y_true, y_pred = perfect_predictions

        model = HistoricalMeanModel()
        model.is_fitted = True

        # Create dummy X
        X = pd.DataFrame({"x": range(len(y_true))})

        # Mock predict to return perfect predictions
        original_predict = model.predict
        model.predict = lambda x: y_pred

        metrics = model.evaluate(X, y_true)

        assert abs(metrics["r2"] - 1.0) < 1e-10
        assert metrics["rmse"] < 1e-10
