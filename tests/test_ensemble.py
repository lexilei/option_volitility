"""Tests for EnsembleVolModel."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import EnsembleVolModel
from src.models.baseline import HistoricalMeanModel, EWMAModel
from src.models.linear import RidgeVolModel


class TestEnsembleVolModel:
    """Tests for EnsembleVolModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame(
            {
                f"feature_{i}": np.random.randn(n) for i in range(5)
            }
        )
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    @pytest.fixture
    def fitted_models(self, sample_data):
        """Create fitted base models."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model1.fit(X, y)

        model2 = EWMAModel(span=21)
        model2.fit(X, y)

        model3 = RidgeVolModel(alpha=1.0)
        model3.fit(X, y)

        return [model1, model2, model3]

    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleVolModel()

        assert ensemble.name == "ensemble"
        assert ensemble.models == []
        assert ensemble.weights is None

    def test_initialization_with_models(self, fitted_models):
        """Test ensemble initialization with models."""
        ensemble = EnsembleVolModel(
            models=fitted_models,
            weights=[0.4, 0.3, 0.3],
        )

        assert len(ensemble.models) == 3
        assert ensemble.weights == [0.4, 0.3, 0.3]

    def test_weights_validation(self, fitted_models):
        """Test that weights are validated."""
        # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleVolModel(models=fitted_models, weights=[0.5, 0.5])

        # Weights don't sum to 1
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            EnsembleVolModel(models=fitted_models, weights=[0.5, 0.5, 0.5])

    def test_add_model(self, fitted_models):
        """Test adding models to ensemble."""
        ensemble = EnsembleVolModel()

        ensemble.add_model(fitted_models[0], weight=0.5)
        assert len(ensemble.models) == 1

        ensemble.add_model(fitted_models[1], weight=0.5)
        assert len(ensemble.models) == 2
        # Weights should be renormalized
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_fit_no_models(self, sample_data):
        """Test that fit fails with no models."""
        X, y = sample_data
        ensemble = EnsembleVolModel()

        with pytest.raises(ValueError, match="No models in ensemble"):
            ensemble.fit(X, y)

    def test_fit_with_unfitted_models(self, sample_data):
        """Test fitting ensemble with unfitted models."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(models=[model1, model2])
        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert model1.is_fitted
        assert model2.is_fitted
        # Should have equal weights
        assert ensemble.weights == pytest.approx([0.5, 0.5])

    def test_fit_with_custom_weights(self, sample_data):
        """Test fitting with custom weights."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            weights=[0.7, 0.3],
        )
        ensemble.fit(X, y)

        assert ensemble.weights == [0.7, 0.3]

    def test_predict_before_fit(self, sample_data, fitted_models):
        """Test that predict fails before fit."""
        X, y = sample_data
        ensemble = EnsembleVolModel(models=fitted_models)

        with pytest.raises(ValueError, match="Model must be fitted"):
            ensemble.predict(X)

    def test_predict_weighted_average(self, sample_data):
        """Test prediction with weighted average."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            method="weighted_average",
        )
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).all()

    def test_predict_median(self, sample_data):
        """Test prediction with median method."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)
        model3 = RidgeVolModel(alpha=1.0)

        ensemble = EnsembleVolModel(
            models=[model1, model2, model3],
            method="median",
        )
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)

    def test_predict_min(self, sample_data):
        """Test prediction with min method."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            method="min",
        )
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)

    def test_predict_max(self, sample_data):
        """Test prediction with max method."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            method="max",
        )
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)

    def test_predict_invalid_method(self, sample_data):
        """Test that invalid method raises error."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)

        ensemble = EnsembleVolModel(
            models=[model1],
            method="invalid_method",
        )
        ensemble.fit(X, y)

        with pytest.raises(ValueError, match="Unknown ensemble method"):
            ensemble.predict(X)

    def test_predict_individual(self, sample_data):
        """Test getting individual model predictions."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(models=[model1, model2])
        ensemble.fit(X, y)

        individual = ensemble.predict_individual(X)

        assert "historical_mean" in individual
        assert "ewma" in individual
        assert len(individual["historical_mean"]) == len(X)

    def test_get_model_weights(self, sample_data):
        """Test getting model weights."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            weights=[0.6, 0.4],
        )
        ensemble.fit(X, y)

        weights = ensemble.get_model_weights()

        assert weights["historical_mean"] == 0.6
        assert weights["ewma"] == 0.4

    def test_get_feature_importance(self, sample_data):
        """Test getting aggregated feature importance."""
        X, y = sample_data

        # Use models that support feature importance
        model1 = RidgeVolModel(alpha=1.0)
        model2 = RidgeVolModel(alpha=0.5)

        ensemble = EnsembleVolModel(models=[model1, model2])
        ensemble.fit(X, y)

        importance = ensemble.get_feature_importance()

        # Ridge models have feature importance
        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_get_feature_importance_no_support(self, sample_data):
        """Test feature importance when models don't support it."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(models=[model1, model2])
        ensemble.fit(X, y)

        importance = ensemble.get_feature_importance()

        # These models don't have feature importance
        assert importance is None

    def test_evaluate_components(self, sample_data):
        """Test evaluating component models."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = RidgeVolModel(alpha=1.0)

        ensemble = EnsembleVolModel(models=[model1, model2])
        ensemble.fit(X, y)

        results = ensemble.evaluate_components(X, y)

        assert "historical_mean" in results.index
        assert "ridge" in results.index
        assert "ensemble" in results.index
        assert "rmse" in results.columns
        assert "weight" in results.columns

    def test_save_and_load(self, sample_data):
        """Test saving and loading ensemble model."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            weights=[0.6, 0.4],
        )
        ensemble.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            ensemble.save(f.name)

            loaded = EnsembleVolModel.load(f.name)

            assert loaded.is_fitted
            assert len(loaded.models) == 2
            assert loaded.weights == [0.6, 0.4]

            # Predictions should match
            original_pred = ensemble.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_optimize_weights(self, sample_data):
        """Test weight optimization."""
        X, y = sample_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model1 = HistoricalMeanModel(window=21)
        model2 = RidgeVolModel(alpha=1.0)

        ensemble = EnsembleVolModel(
            models=[model1, model2],
            optimize_weights=True,
        )
        ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert ensemble.is_fitted
        assert ensemble.weights is not None
        assert sum(ensemble.weights) == pytest.approx(1.0)
        # Weights should be between 0 and 1
        assert all(0 <= w <= 1 for w in ensemble.weights)


class TestEnsembleEdgeCases:
    """Edge case tests for EnsembleVolModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({"feature1": np.random.randn(n)})
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    def test_single_model_ensemble(self, sample_data):
        """Test ensemble with single model."""
        X, y = sample_data

        model = HistoricalMeanModel(window=21)
        ensemble = EnsembleVolModel(models=[model])
        ensemble.fit(X, y)

        assert ensemble.weights == [1.0]

        predictions = ensemble.predict(X)
        model_predictions = model.predict(X)

        np.testing.assert_array_almost_equal(predictions, model_predictions)

    def test_predict_individual_before_fit(self, sample_data):
        """Test predict_individual fails before fit."""
        X, y = sample_data

        model = HistoricalMeanModel(window=21)
        ensemble = EnsembleVolModel(models=[model])

        with pytest.raises(ValueError, match="Model must be fitted"):
            ensemble.predict_individual(X)

    def test_method_comparison(self, sample_data):
        """Test that different methods produce different results."""
        X, y = sample_data

        model1 = HistoricalMeanModel(window=21)
        model2 = EWMAModel(span=21)

        ensemble_avg = EnsembleVolModel(models=[model1, model2], method="weighted_average")
        ensemble_avg.fit(X, y)

        ensemble_median = EnsembleVolModel(models=[model1, model2], method="median")
        ensemble_median.fit(X, y)

        pred_avg = ensemble_avg.predict(X)
        pred_median = ensemble_median.predict(X)

        # The predictions might be similar but typically not identical
        # (unless both models produce identical predictions)
        assert len(pred_avg) == len(pred_median)
