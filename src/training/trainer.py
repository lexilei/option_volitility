"""Unified training interface for volatility models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseVolModel
from src.models.baseline import HistoricalMeanModel, NaiveModel, EWMAModel
from src.models.linear import RidgeVolModel, LassoVolModel, ElasticNetVolModel
from src.models.tree import XGBoostVolModel, LightGBMVolModel, RandomForestVolModel
from src.models.lstm import LSTMVolModel
from src.models.tft import TFTVolModel
from src.models.ensemble import EnsembleVolModel
from src.training.walk_forward import WalkForwardCV, WalkForwardResult
from src.training.hyperopt import HyperoptTuner

if TYPE_CHECKING:
    pass


# Model registry
MODEL_REGISTRY: dict[str, type[BaseVolModel]] = {
    "historical_mean": HistoricalMeanModel,
    "naive": NaiveModel,
    "ewma": EWMAModel,
    "ridge": RidgeVolModel,
    "lasso": LassoVolModel,
    "elasticnet": ElasticNetVolModel,
    "xgboost": XGBoostVolModel,
    "lightgbm": LightGBMVolModel,
    "random_forest": RandomForestVolModel,
    "lstm": LSTMVolModel,
    "tft": TFTVolModel,
    "ensemble": EnsembleVolModel,
}


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str
    model_params: dict[str, Any] = field(default_factory=dict)
    use_walk_forward: bool = True
    train_window_days: int = 504
    test_window_days: int = 63
    step_days: int = 63
    tune_hyperparams: bool = False
    n_trials: int = 50
    early_stopping_rounds: int | None = None
    save_model: bool = True
    save_dir: str = "data/models"


@dataclass
class TrainingResult:
    """Results from model training."""

    model_name: str
    config: TrainingConfig
    cv_results: list[WalkForwardResult] | None
    aggregate_metrics: dict[str, float]
    final_model: BaseVolModel
    best_params: dict[str, Any] | None
    feature_importance: pd.Series | None
    model_path: str | None = None


class ModelTrainer:
    """Unified trainer for volatility prediction models."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ):
        """Initialize the trainer.

        Args:
            X: Training features with datetime index
            y: Training targets with datetime index
            X_test: Optional held-out test features
            y_test: Optional held-out test targets
        """
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

        self.results: dict[str, TrainingResult] = {}

    def get_model_class(self, model_name: str) -> type[BaseVolModel]:
        """Get model class by name.

        Args:
            model_name: Name of the model

        Returns:
            Model class
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model_name]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a single model.

        Args:
            config: Training configuration

        Returns:
            TrainingResult with model and metrics
        """
        logger.info(f"Training {config.model_name}")

        model_class = self.get_model_class(config.model_name)
        model_params = config.model_params.copy()
        best_params = None

        # Hyperparameter tuning
        if config.tune_hyperparams and config.model_name not in ["historical_mean", "naive", "ewma"]:
            logger.info(f"Tuning hyperparameters for {config.model_name}")
            tuner = HyperoptTuner(
                model_class=model_class,
                n_trials=config.n_trials,
                cv=WalkForwardCV(
                    train_window_days=config.train_window_days,
                    test_window_days=config.test_window_days,
                    step_days=config.step_days,
                ) if config.use_walk_forward else None,
            )
            best_params = tuner.tune(self.X, self.y)
            model_params.update(best_params)

        # Walk-forward cross-validation
        cv_results = None
        aggregate_metrics = {}

        if config.use_walk_forward:
            cv = WalkForwardCV(
                train_window_days=config.train_window_days,
                test_window_days=config.test_window_days,
                step_days=config.step_days,
            )

            cv_results = cv.cross_validate(
                model_class,
                self.X,
                self.y,
                model_params=model_params,
                return_models=False,
            )

            aggregate_metrics = cv.get_aggregate_metrics(cv_results)
            logger.info(f"CV Results - RMSE: {aggregate_metrics['overall_rmse']:.4f}, R2: {aggregate_metrics['overall_r2']:.4f}")
        else:
            # Simple train/test split
            if self.X_test is not None and self.y_test is not None:
                model = model_class(**model_params)
                model.fit(self.X, self.y)
                aggregate_metrics = model.evaluate(self.X_test, self.y_test)
            else:
                # Use last 20% as validation
                split_idx = int(len(self.X) * 0.8)
                X_train, X_val = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
                y_train, y_val = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

                model = model_class(**model_params)
                model.fit(X_train, y_train)
                aggregate_metrics = model.evaluate(X_val, y_val)

        # Train final model on all data
        final_model = model_class(**model_params)
        final_model.fit(self.X, self.y)

        # Feature importance
        feature_importance = final_model.get_feature_importance()

        # Save model
        model_path = None
        if config.save_model:
            save_dir = Path(config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(save_dir / f"{config.model_name}.joblib")
            final_model.save(model_path)

        result = TrainingResult(
            model_name=config.model_name,
            config=config,
            cv_results=cv_results,
            aggregate_metrics=aggregate_metrics,
            final_model=final_model,
            best_params=best_params,
            feature_importance=feature_importance,
            model_path=model_path,
        )

        self.results[config.model_name] = result
        return result

    def train_multiple(
        self,
        model_names: list[str] | None = None,
        common_config: dict[str, Any] | None = None,
        model_configs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, TrainingResult]:
        """Train multiple models.

        Args:
            model_names: List of model names to train (None for all)
            common_config: Configuration shared by all models
            model_configs: Model-specific configuration overrides

        Returns:
            Dictionary of TrainingResult objects
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())
            # Exclude ensemble as it requires separate handling
            model_names.remove("ensemble")

        if common_config is None:
            common_config = {}

        if model_configs is None:
            model_configs = {}

        results = {}
        for name in model_names:
            try:
                # Merge configurations
                config_dict = {
                    "model_name": name,
                    **common_config,
                    **model_configs.get(name, {}),
                }
                config = TrainingConfig(**config_dict)

                result = self.train(config)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        return results

    def train_ensemble(
        self,
        base_model_names: list[str] | None = None,
        optimize_weights: bool = True,
        save_model: bool = True,
        save_dir: str = "data/models",
    ) -> TrainingResult:
        """Train an ensemble of models.

        Args:
            base_model_names: Models to include in ensemble
            optimize_weights: Whether to optimize ensemble weights
            save_model: Whether to save the ensemble
            save_dir: Directory to save model

        Returns:
            TrainingResult for the ensemble
        """
        if base_model_names is None:
            base_model_names = ["ridge", "xgboost", "lightgbm"]

        # Train base models if not already trained
        for name in base_model_names:
            if name not in self.results:
                config = TrainingConfig(model_name=name)
                self.train(config)

        # Create ensemble
        base_models = [self.results[name].final_model for name in base_model_names]

        ensemble = EnsembleVolModel(
            models=base_models,
            optimize_weights=optimize_weights,
        )

        # Validation set for weight optimization
        split_idx = int(len(self.X) * 0.8)
        X_train, X_val = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
        y_train, y_val = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

        ensemble.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        aggregate_metrics = ensemble.evaluate(X_val, y_val)

        # Save
        model_path = None
        if save_model:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            model_path = str(save_dir_path / "ensemble.joblib")
            ensemble.save(model_path)

        config = TrainingConfig(
            model_name="ensemble",
            save_model=save_model,
            save_dir=save_dir,
        )

        result = TrainingResult(
            model_name="ensemble",
            config=config,
            cv_results=None,
            aggregate_metrics=aggregate_metrics,
            final_model=ensemble,
            best_params={"weights": ensemble.weights, "base_models": base_model_names},
            feature_importance=ensemble.get_feature_importance(),
            model_path=model_path,
        )

        self.results["ensemble"] = result
        return result

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models.

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        records = []
        for name, result in self.results.items():
            record = {
                "model": name,
                **result.aggregate_metrics,
            }
            records.append(record)

        df = pd.DataFrame(records).set_index("model")

        # Sort by RMSE (or R2 if available)
        if "overall_rmse" in df.columns:
            df = df.sort_values("overall_rmse")
        elif "rmse" in df.columns:
            df = df.sort_values("rmse")

        return df

    def get_best_model(self, metric: str = "rmse") -> BaseVolModel:
        """Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Best performing model
        """
        comparison = self.compare_models()

        # Find metric column
        metric_col = metric if metric in comparison.columns else f"overall_{metric}"
        if metric_col not in comparison.columns:
            raise ValueError(f"Metric {metric} not found in results")

        best_model_name = comparison[metric_col].idxmin()
        return self.results[best_model_name].final_model

    def get_predictions_df(self) -> pd.DataFrame:
        """Get predictions from all models as a DataFrame.

        Returns:
            DataFrame with predictions from each model
        """
        predictions = {}

        for name, result in self.results.items():
            predictions[name] = result.final_model.predict(self.X)

        df = pd.DataFrame(predictions, index=self.X.index)
        df["actual"] = self.y.values

        return df


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: str = "data/models",
    tune_hyperparams: bool = False,
    n_trials: int = 50,
) -> dict[str, TrainingResult]:
    """Convenience function to train all available models.

    Args:
        X: Features with datetime index
        y: Targets with datetime index
        save_dir: Directory to save models
        tune_hyperparams: Whether to tune hyperparameters
        n_trials: Number of tuning trials

    Returns:
        Dictionary of TrainingResult objects
    """
    trainer = ModelTrainer(X, y)

    # Train individual models
    common_config = {
        "use_walk_forward": True,
        "tune_hyperparams": tune_hyperparams,
        "n_trials": n_trials,
        "save_model": True,
        "save_dir": save_dir,
    }

    results = trainer.train_multiple(
        model_names=["historical_mean", "ridge", "lasso", "xgboost", "lightgbm"],
        common_config=common_config,
    )

    # Train ensemble
    trainer.train_ensemble(
        base_model_names=["ridge", "xgboost", "lightgbm"],
        save_dir=save_dir,
    )

    # Compare
    comparison = trainer.compare_models()
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison.to_string()}")

    return trainer.results
