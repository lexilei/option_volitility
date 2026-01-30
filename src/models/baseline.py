"""Baseline models for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseVolModel


class HistoricalMeanModel(BaseVolModel):
    """Historical mean baseline model.

    Predicts future volatility as the rolling mean of past volatility.
    """

    name = "historical_mean"

    def __init__(
        self,
        window: int = 21,
        min_periods: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the model.

        Args:
            window: Rolling window size for historical mean
            min_periods: Minimum periods required (default: window // 2)
            **kwargs: Additional parameters
        """
        super().__init__(window=window, min_periods=min_periods, **kwargs)
        self.window = window
        self.min_periods = min_periods or window // 2
        self._last_values: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "HistoricalMeanModel":
        """Fit the model (store the last window values for prediction).

        Args:
            X: Feature matrix (not used, but kept for API consistency)
            y: Target values (historical volatility)
            **kwargs: Additional arguments (ignored)

        Returns:
            Self
        """
        y_arr = self._validate_target(y)

        # Store the last window values for prediction
        self._last_values = y_arr[-self.window :]

        self.is_fitted = True
        logger.info(f"Fitted {self.name} with window={self.window}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict future volatility.

        For in-sample data, uses rolling mean.
        For out-of-sample, uses the mean of stored historical values.

        Args:
            X: Feature matrix (used to determine number of predictions)

        Returns:
            Predicted volatility values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        # Return the historical mean for all predictions
        if self._last_values is not None and len(self._last_values) > 0:
            mean_val = np.mean(self._last_values)
        else:
            mean_val = np.nan

        return np.full(n_samples, mean_val)

    def predict_rolling(
        self,
        y: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Predict using rolling mean (for in-sample evaluation).

        Args:
            y: Historical volatility values

        Returns:
            Rolling mean predictions
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        return y.rolling(window=self.window, min_periods=self.min_periods).mean().values


class NaiveModel(BaseVolModel):
    """Naive baseline model.

    Predicts future volatility as the last observed value.
    """

    name = "naive"

    def __init__(self, **kwargs: Any):
        """Initialize the model.

        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self._last_value: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "NaiveModel":
        """Fit the model (store the last value).

        Args:
            X: Feature matrix (not used)
            y: Target values
            **kwargs: Additional arguments

        Returns:
            Self
        """
        y_arr = self._validate_target(y)
        self._last_value = y_arr[-1]
        self.is_fitted = True
        logger.info(f"Fitted {self.name}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict using last observed value.

        Args:
            X: Feature matrix (used to determine number of predictions)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        return np.full(n_samples, self._last_value)


class SeasonalNaiveModel(BaseVolModel):
    """Seasonal naive baseline model.

    Predicts future volatility based on values from the same period in the past.
    """

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 21, **kwargs: Any):
        """Initialize the model.

        Args:
            seasonal_period: Number of periods in a season (default: 21 trading days)
            **kwargs: Additional parameters
        """
        super().__init__(seasonal_period=seasonal_period, **kwargs)
        self.seasonal_period = seasonal_period
        self._seasonal_values: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "SeasonalNaiveModel":
        """Fit the model (store seasonal values).

        Args:
            X: Feature matrix (not used)
            y: Target values
            **kwargs: Additional arguments

        Returns:
            Self
        """
        y_arr = self._validate_target(y)

        # Store the last seasonal_period values
        self._seasonal_values = y_arr[-self.seasonal_period :]

        self.is_fitted = True
        logger.info(f"Fitted {self.name} with period={self.seasonal_period}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict using seasonal pattern.

        Args:
            X: Feature matrix (used to determine number of predictions)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        # Tile seasonal values to cover all predictions
        n_tiles = (n_samples // self.seasonal_period) + 1
        predictions = np.tile(self._seasonal_values, n_tiles)[:n_samples]

        return predictions


class EWMAModel(BaseVolModel):
    """Exponentially Weighted Moving Average model.

    Predicts future volatility using EWMA.
    """

    name = "ewma"

    def __init__(self, span: int = 21, **kwargs: Any):
        """Initialize the model.

        Args:
            span: EWMA span
            **kwargs: Additional parameters
        """
        super().__init__(span=span, **kwargs)
        self.span = span
        self._ewma_value: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "EWMAModel":
        """Fit the model (compute EWMA of training data).

        Args:
            X: Feature matrix (not used)
            y: Target values
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        ewma = y.ewm(span=self.span, adjust=False).mean()
        self._ewma_value = ewma.iloc[-1]

        self.is_fitted = True
        logger.info(f"Fitted {self.name} with span={self.span}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict using EWMA.

        Args:
            X: Feature matrix (used to determine number of predictions)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        return np.full(n_samples, self._ewma_value)
