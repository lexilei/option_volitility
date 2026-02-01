"""Tests for training modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.walk_forward import WalkForwardCV, ExpandingWindowCV

# Check if optuna is available
try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class TestWalkForwardCV:
    """Tests for WalkForwardCV class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 756  # 3 years

        dates = pd.date_range(start="2021-01-01", periods=n, freq="B")
        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        }, index=dates)
        y = pd.Series(np.random.uniform(0.1, 0.3, n), index=dates)

        return X, y

    @pytest.fixture
    def cv(self):
        """Create a walk-forward CV instance."""
        return WalkForwardCV(
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )

    def test_split_generates_folds(self, cv, sample_data):
        """Test that split generates folds."""
        X, y = sample_data
        folds = list(cv.split(X))

        assert len(folds) > 0
        for fold in folds:
            assert len(fold.train_indices) > 0
            assert len(fold.test_indices) > 0

    def test_no_data_leakage(self, cv, sample_data):
        """Test that train data comes before test data."""
        X, y = sample_data
        dates = pd.DatetimeIndex(X.index)

        for fold in cv.split(X):
            train_end = dates[fold.train_indices[-1]]
            test_start = dates[fold.test_indices[0]]
            assert train_end < test_start

    def test_get_n_splits(self, cv, sample_data):
        """Test getting number of splits."""
        X, y = sample_data
        n_splits = cv.get_n_splits(X)

        assert n_splits > 0
        assert n_splits == len(list(cv.split(X)))

    def test_cross_validate(self, cv, sample_data):
        """Test cross-validation with a simple model."""
        from src.models.baseline import HistoricalMeanModel

        X, y = sample_data

        results = cv.cross_validate(
            HistoricalMeanModel,
            X,
            y,
            model_params={"window": 21},
            verbose=False,
        )

        assert len(results) > 0
        for result in results:
            assert "rmse" in result.metrics
            assert len(result.predictions) == len(result.actuals)

    def test_get_aggregate_metrics(self, cv, sample_data):
        """Test aggregate metrics calculation."""
        from src.models.baseline import HistoricalMeanModel

        X, y = sample_data

        results = cv.cross_validate(
            HistoricalMeanModel,
            X,
            y,
            model_params={"window": 21},
            verbose=False,
        )

        agg_metrics = cv.get_aggregate_metrics(results)

        assert "overall_rmse" in agg_metrics
        assert "overall_r2" in agg_metrics
        assert "n_folds" in agg_metrics
        assert agg_metrics["n_folds"] == len(results)

    def test_results_to_dataframe(self, cv, sample_data):
        """Test converting results to DataFrame."""
        from src.models.baseline import HistoricalMeanModel

        X, y = sample_data

        results = cv.cross_validate(
            HistoricalMeanModel,
            X,
            y,
            model_params={"window": 21},
            verbose=False,
        )

        df = cv.results_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert "fold_id" in df.columns
        assert "rmse" in df.columns


class TestExpandingWindowCV:
    """Tests for ExpandingWindowCV class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 756

        dates = pd.date_range(start="2021-01-01", periods=n, freq="B")
        X = pd.DataFrame({"feature1": np.random.randn(n)}, index=dates)
        y = pd.Series(np.random.uniform(0.1, 0.3, n), index=dates)

        return X, y

    @pytest.fixture
    def cv(self):
        """Create an expanding window CV instance."""
        return ExpandingWindowCV(
            train_window_days=252,
            test_window_days=63,
            step_days=63,
        )

    def test_training_window_expands(self, cv, sample_data):
        """Test that training window expands over folds."""
        X, y = sample_data
        folds = list(cv.split(X))

        train_sizes = [len(fold.train_indices) for fold in folds]

        # Each fold should have equal or larger training set
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_first_fold_starts_at_zero(self, cv, sample_data):
        """Test that first fold starts from beginning."""
        X, y = sample_data
        folds = list(cv.split(X))

        assert folds[0].train_indices[0] == 0


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
class TestHyperoptTuner:
    """Tests for HyperoptTuner class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        })
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    def test_import(self):
        """Test that HyperoptTuner can be imported."""
        from src.training.hyperopt import HyperoptTuner, HAS_OPTUNA

        assert HyperoptTuner is not None
        assert HAS_OPTUNA is True  # Should be True if we get here (class is skipped otherwise)

    def test_initialization(self):
        """Test tuner initialization."""
        from src.training.hyperopt import HyperoptTuner
        from src.models.linear import RidgeVolModel

        tuner = HyperoptTuner(
            model_class=RidgeVolModel,
            n_trials=5,
        )

        assert tuner.n_trials == 5
        assert tuner.model_class == RidgeVolModel

    def test_tune_quick(self, sample_data):
        """Test tuning with few trials."""
        from src.training.hyperopt import HyperoptTuner
        from src.models.linear import RidgeVolModel

        X, y = sample_data

        tuner = HyperoptTuner(
            model_class=RidgeVolModel,
            n_trials=3,  # Very few trials for speed
        )

        best_params = tuner.tune(X, y, verbose=False)

        assert "alpha" in best_params
        assert tuner.best_score is not None


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        n = 300

        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        X = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
        }, index=dates)
        y = pd.Series(np.random.uniform(0.1, 0.3, n), index=dates, name="target")

        return X, y

    def test_import(self):
        """Test that ModelTrainer can be imported."""
        from src.training.trainer import ModelTrainer

        assert ModelTrainer is not None

    def test_initialization(self, sample_data):
        """Test trainer initialization."""
        from src.training.trainer import ModelTrainer

        X, y = sample_data
        trainer = ModelTrainer(X, y)

        assert trainer.X is not None
        assert trainer.y is not None

    def test_get_model_class(self, sample_data):
        """Test getting model class by name."""
        from src.training.trainer import ModelTrainer
        from src.models.baseline import HistoricalMeanModel

        X, y = sample_data
        trainer = ModelTrainer(X, y)

        model_class = trainer.get_model_class("historical_mean")
        assert model_class == HistoricalMeanModel

    def test_train_baseline(self, sample_data):
        """Test training baseline model."""
        from src.training.trainer import ModelTrainer, TrainingConfig

        X, y = sample_data
        trainer = ModelTrainer(X, y)

        config = TrainingConfig(
            model_name="historical_mean",
            use_walk_forward=False,
            save_model=False,
        )

        result = trainer.train(config)

        assert result.final_model is not None
        assert result.final_model.is_fitted
