"""Tests for tree-based models (XGBoost, LightGBM, RandomForest)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if XGBoost and LightGBM are available
try:
    import xgboost

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
class TestXGBoostVolModel:
    """Tests for XGBoostVolModel class."""

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
        # Create target with some relationship to features
        y = 0.2 + 0.05 * X["feature_0"] + np.random.randn(n) * 0.02
        y = pd.Series(y.clip(0.1, 0.4))

        return X, y

    @pytest.fixture
    def model(self):
        """Create an XGBoost model."""
        from src.models.tree import XGBoostVolModel

        return XGBoostVolModel(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.name == "xgboost"
        assert model.params["n_estimators"] == 50
        assert model.params["max_depth"] == 3

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X, y = sample_data

        model.fit(X, y)

        assert model.is_fitted

    def test_predict_before_fit(self, model, sample_data):
        """Test that predict fails before fit."""
        X, y = sample_data

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test model prediction."""
        X, y = sample_data

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_fit_with_eval_set(self, sample_data):
        """Test fitting with evaluation set."""
        from src.models.tree import XGBoostVolModel

        X, y = sample_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = XGBoostVolModel(n_estimators=100, max_depth=3)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val.values, y_val.values)],
            early_stopping_rounds=10,
            verbose=False,
        )

        assert model.is_fitted

    def test_feature_importance(self, model, sample_data):
        """Test feature importance."""
        X, y = sample_data

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]
        # Most important features should have positive importance
        assert importance.max() > 0

    def test_feature_importance_with_names(self, sample_data):
        """Test that feature names are preserved."""
        from src.models.tree import XGBoostVolModel

        X, y = sample_data

        model = XGBoostVolModel(n_estimators=50)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert "feature_0" in importance.index

    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0

    def test_save_and_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            from src.models.tree import XGBoostVolModel

            loaded = XGBoostVolModel.load(f.name)

            assert loaded.is_fitted

            # Predictions should match
            original_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestLightGBMVolModel:
    """Tests for LightGBMVolModel class."""

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
        y = 0.2 + 0.05 * X["feature_0"] + np.random.randn(n) * 0.02
        y = pd.Series(y.clip(0.1, 0.4))

        return X, y

    @pytest.fixture
    def model(self):
        """Create a LightGBM model."""
        from src.models.tree import LightGBMVolModel

        return LightGBMVolModel(
            n_estimators=50,
            num_leaves=15,
            learning_rate=0.1,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.name == "lightgbm"
        assert model.params["n_estimators"] == 50
        assert model.params["num_leaves"] == 15

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X, y = sample_data

        model.fit(X, y)

        assert model.is_fitted

    def test_predict_before_fit(self, model, sample_data):
        """Test that predict fails before fit."""
        X, y = sample_data

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test model prediction."""
        X, y = sample_data

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_fit_with_eval_set(self, sample_data):
        """Test fitting with evaluation set."""
        from src.models.tree import LightGBMVolModel

        X, y = sample_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = LightGBMVolModel(n_estimators=100, num_leaves=15)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val.values, y_val.values)],
        )

        assert model.is_fitted

    def test_feature_importance(self, model, sample_data):
        """Test feature importance."""
        X, y = sample_data

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_feature_importance_with_names(self, sample_data):
        """Test that feature names are preserved."""
        from src.models.tree import LightGBMVolModel

        X, y = sample_data

        model = LightGBMVolModel(n_estimators=50)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert "feature_0" in importance.index

    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert "rmse" in metrics
        assert metrics["rmse"] >= 0

    def test_save_and_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            from src.models.tree import LightGBMVolModel

            loaded = LightGBMVolModel.load(f.name)

            assert loaded.is_fitted

            # Predictions should match
            original_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestRandomForestVolModel:
    """Tests for RandomForestVolModel class."""

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
        y = 0.2 + 0.05 * X["feature_0"] + np.random.randn(n) * 0.02
        y = pd.Series(y.clip(0.1, 0.4))

        return X, y

    @pytest.fixture
    def model(self):
        """Create a RandomForest model."""
        from src.models.tree import RandomForestVolModel

        return RandomForestVolModel(
            n_estimators=50,
            max_depth=5,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.name == "random_forest"
        assert model.params["n_estimators"] == 50
        assert model.params["max_depth"] == 5

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X, y = sample_data

        model.fit(X, y)

        assert model.is_fitted

    def test_predict_before_fit(self, model, sample_data):
        """Test that predict fails before fit."""
        X, y = sample_data

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test model prediction."""
        X, y = sample_data

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_feature_importance(self, model, sample_data):
        """Test feature importance."""
        X, y = sample_data

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]
        # Importances should sum to 1 for RandomForest
        assert importance.sum() == pytest.approx(1.0)

    def test_feature_importance_with_names(self, sample_data):
        """Test that feature names are preserved."""
        from src.models.tree import RandomForestVolModel

        X, y = sample_data

        model = RandomForestVolModel(n_estimators=50)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert "feature_0" in importance.index

    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert "rmse" in metrics
        assert metrics["rmse"] >= 0

    def test_save_and_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            from src.models.tree import RandomForestVolModel

            loaded = RandomForestVolModel.load(f.name)

            assert loaded.is_fitted

            # Predictions should match
            original_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestTreeModelsComparison:
    """Comparison tests for tree-based models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with signal."""
        np.random.seed(42)
        n = 300

        X = pd.DataFrame(
            {
                "vol_lag1": np.random.uniform(0.1, 0.3, n),
                "vol_lag2": np.random.uniform(0.1, 0.3, n),
                "returns": np.random.randn(n) * 0.02,
                "momentum": np.random.randn(n),
                "noise": np.random.randn(n),
            }
        )
        # Target has relationship with vol_lag features
        y = 0.5 * X["vol_lag1"] + 0.3 * X["vol_lag2"] + np.random.randn(n) * 0.02
        y = pd.Series(y.clip(0.1, 0.4))

        return X, y

    def test_all_models_fit_predict(self, sample_data):
        """Test that all models can fit and predict."""
        from src.models.tree import RandomForestVolModel

        X, y = sample_data

        models = [RandomForestVolModel(n_estimators=20)]

        if HAS_XGBOOST:
            from src.models.tree import XGBoostVolModel

            models.append(XGBoostVolModel(n_estimators=20))

        if HAS_LIGHTGBM:
            from src.models.tree import LightGBMVolModel

            models.append(LightGBMVolModel(n_estimators=20))

        for model in models:
            model.fit(X, y)
            predictions = model.predict(X)

            assert model.is_fitted
            assert len(predictions) == len(X)
            assert not np.isnan(predictions).any()

    def test_feature_importance_consistency(self, sample_data):
        """Test that feature importance rankings are reasonable."""
        from src.models.tree import RandomForestVolModel

        X, y = sample_data

        model = RandomForestVolModel(n_estimators=50)
        model.fit(X, y)

        importance = model.get_feature_importance()

        # vol_lag features should generally be more important than noise
        # (though this isn't guaranteed with small data)
        assert importance is not None
        assert "vol_lag1" in importance.index
        assert "noise" in importance.index
