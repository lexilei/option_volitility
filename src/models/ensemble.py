"""Ensemble model for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseVolModel


class EnsembleVolModel(BaseVolModel):
    """Ensemble model combining multiple volatility models."""

    name = "ensemble"

    def __init__(
        self,
        models: list[BaseVolModel] | None = None,
        weights: list[float] | None = None,
        method: str = "weighted_average",
        optimize_weights: bool = False,
        **kwargs: Any,
    ):
        """Initialize the ensemble model.

        Args:
            models: List of models to ensemble
            weights: Weights for each model (must sum to 1)
            method: Ensemble method ('weighted_average', 'median', 'min', 'max')
            optimize_weights: Whether to optimize weights based on validation performance
            **kwargs: Additional parameters
        """
        super().__init__(
            method=method,
            optimize_weights=optimize_weights,
            **kwargs,
        )
        self.models = models or []
        self.weights = weights
        self.method = method
        self.optimize_weights = optimize_weights

        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1")

    def add_model(self, model: BaseVolModel, weight: float | None = None) -> "EnsembleVolModel":
        """Add a model to the ensemble.

        Args:
            model: Model to add
            weight: Weight for the model (optional)

        Returns:
            Self
        """
        self.models.append(model)

        if weight is not None:
            if self.weights is None:
                self.weights = [weight]
            else:
                self.weights.append(weight)
                # Renormalize weights
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]

        return self

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> "EnsembleVolModel":
        """Fit all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target values
            X_val: Validation features (for weight optimization)
            y_val: Validation targets
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        # Fit each model
        for model in self.models:
            logger.info(f"Fitting {model.name} in ensemble")
            model.fit(X_arr, y_arr, **kwargs)

        # Optimize weights if requested
        if self.optimize_weights and X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
        elif self.weights is None:
            # Equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self.is_fitted = True
        logger.info(
            f"Fitted ensemble with {len(self.models)} models, "
            f"weights={[f'{w:.3f}' for w in self.weights]}"
        )
        return self

    def _optimize_weights(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> None:
        """Optimize ensemble weights based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        from scipy.optimize import minimize

        X_val_arr = self._validate_input(X_val)
        y_val_arr = self._validate_target(y_val)

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X_val_arr)
            predictions.append(pred)
        predictions = np.array(predictions)  # (n_models, n_samples)

        # Remove NaN samples
        valid_mask = ~np.isnan(predictions).any(axis=0) & ~np.isnan(y_val_arr)
        predictions = predictions[:, valid_mask]
        y_valid = y_val_arr[valid_mask]

        def objective(weights: np.ndarray) -> float:
            """MSE objective function."""
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return np.mean((y_valid - ensemble_pred) ** 2)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: weights between 0 and 1
        bounds = [(0, 1)] * len(self.models)

        # Initial guess: equal weights
        x0 = np.ones(len(self.models)) / len(self.models)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            self.weights = result.x.tolist()
            logger.info(f"Optimized weights: {[f'{w:.3f}' for w in self.weights]}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X_arr)
            predictions.append(pred)
        predictions = np.array(predictions)  # (n_models, n_samples)

        # Combine predictions
        if self.method == "weighted_average":
            return np.average(predictions, axis=0, weights=self.weights)
        elif self.method == "median":
            return np.median(predictions, axis=0)
        elif self.method == "min":
            return np.min(predictions, axis=0)
        elif self.method == "max":
            return np.max(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_individual(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Get predictions from each individual model.

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping model name to predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)

        results = {}
        for model in self.models:
            results[model.name] = model.predict(X_arr)

        return results

    def get_feature_importance(self) -> pd.Series | None:
        """Get aggregated feature importance from all models.

        Returns:
            Series with feature importance (averaged across models)
        """
        if not self.is_fitted:
            return None

        all_importance = []
        model_weights = []

        for model, weight in zip(self.models, self.weights):
            importance = model.get_feature_importance()
            if importance is not None:
                all_importance.append(importance)
                model_weights.append(weight)

        if not all_importance:
            return None

        # Align indices and compute weighted average
        combined = pd.DataFrame(all_importance).T
        combined.columns = [m.name for m in self.models if m.get_feature_importance() is not None]

        # Normalize weights
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()

        # Weighted average
        weighted_avg = (combined * model_weights).sum(axis=1)
        return weighted_avg.sort_values(ascending=False)

    def get_model_weights(self) -> dict[str, float]:
        """Get weights for each model.

        Returns:
            Dictionary mapping model name to weight
        """
        return {model.name: weight for model, weight in zip(self.models, self.weights)}

    def evaluate_components(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate each component model individually.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            DataFrame with metrics for each model
        """
        results = []
        for model in self.models:
            metrics = model.evaluate(X, y)
            metrics["model"] = model.name
            metrics["weight"] = self.weights[self.models.index(model)]
            results.append(metrics)

        # Add ensemble metrics
        ensemble_metrics = self.evaluate(X, y)
        ensemble_metrics["model"] = "ensemble"
        ensemble_metrics["weight"] = 1.0
        results.append(ensemble_metrics)

        return pd.DataFrame(results).set_index("model")

    def save(self, path: str) -> None:
        """Save the ensemble model.

        Args:
            path: Path to save the model
        """
        import joblib
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "params": self.params,
            "is_fitted": self.is_fitted,
            "weights": self.weights,
            "method": self.method,
            "models": self.models,  # Each model should be serializable
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved ensemble model to {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleVolModel":
        """Load an ensemble model.

        Args:
            path: Path to the saved model

        Returns:
            Loaded ensemble model
        """
        import joblib

        model_data = joblib.load(path)

        instance = cls(
            models=model_data["models"],
            weights=model_data["weights"],
            method=model_data["method"],
        )
        instance.is_fitted = model_data["is_fitted"]

        logger.info(f"Loaded ensemble model from {path}")
        return instance
