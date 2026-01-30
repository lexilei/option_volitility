"""Base class for volatility prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class BaseVolModel(ABC):
    """Abstract base class for volatility prediction models."""

    name: str = "base"

    def __init__(self, **kwargs: Any):
        """Initialize the model.

        Args:
            **kwargs: Model-specific parameters
        """
        self.params = kwargs
        self.is_fitted = False
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "BaseVolModel":
        """Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        pass

    def fit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Fit and predict in one step.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Predicted values
        """
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "name": self.name,
        }
        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseVolModel":
        """Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model_data["model"]

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseVolModel":
        """Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self
        """
        self.params.update(params)
        return self

    def _validate_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Validate and convert input to numpy array.

        Args:
            X: Input data

        Returns:
            Numpy array
        """
        if isinstance(X, pd.DataFrame):
            if not self.is_fitted:
                self.feature_names = list(X.columns)
            return X.values
        return np.asarray(X)

    def _validate_target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Validate and convert target to numpy array.

        Args:
            y: Target data

        Returns:
            Numpy array
        """
        if isinstance(y, pd.Series):
            return y.values
        return np.asarray(y)

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance if available.

        Returns:
            Series with feature importance or None
        """
        return None

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_true = self._validate_target(y)

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE with protection against division by zero
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            mape = np.nan

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, is_fitted={self.is_fitted})"
