"""Linear models for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .base import BaseVolModel


class RidgeVolModel(BaseVolModel):
    """Ridge regression model for volatility prediction."""

    name = "ridge"

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Ridge model.

        Args:
            alpha: Regularization strength
            fit_intercept: Whether to fit intercept
            normalize_features: Whether to standardize features
            **kwargs: Additional parameters passed to sklearn Ridge
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize_features=normalize_features,
            **kwargs,
        )
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features

        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self.scaler = StandardScaler() if normalize_features else None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "RidgeVolModel":
        """Fit the Ridge model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        # Standardize features
        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

        logger.info(f"Fitted {self.name} with alpha={self.alpha}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)

        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        return self.model.predict(X_arr)

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature coefficients as importance.

        Returns:
            Series with absolute coefficients
        """
        if not self.is_fitted:
            return None

        importance = np.abs(self.model.coef_)
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)

    def get_coefficients(self) -> pd.Series | None:
        """Get model coefficients.

        Returns:
            Series with coefficients
        """
        if not self.is_fitted:
            return None

        if self.feature_names:
            return pd.Series(self.model.coef_, index=self.feature_names)
        return pd.Series(self.model.coef_)


class LassoVolModel(BaseVolModel):
    """Lasso regression model for volatility prediction."""

    name = "lasso"

    def __init__(
        self,
        alpha: float = 0.1,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        max_iter: int = 10000,
        **kwargs: Any,
    ):
        """Initialize the Lasso model.

        Args:
            alpha: Regularization strength
            fit_intercept: Whether to fit intercept
            normalize_features: Whether to standardize features
            max_iter: Maximum iterations
            **kwargs: Additional parameters
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize_features=normalize_features,
            max_iter=max_iter,
            **kwargs,
        )
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.max_iter = max_iter

        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        self.scaler = StandardScaler() if normalize_features else None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "LassoVolModel":
        """Fit the Lasso model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

        # Count non-zero coefficients
        n_nonzero = np.sum(self.model.coef_ != 0)
        logger.info(f"Fitted {self.name} with alpha={self.alpha}, {n_nonzero} non-zero coefficients")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)

        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        return self.model.predict(X_arr)

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature coefficients as importance.

        Returns:
            Series with absolute coefficients (non-zero only)
        """
        if not self.is_fitted:
            return None

        importance = np.abs(self.model.coef_)
        if self.feature_names:
            series = pd.Series(importance, index=self.feature_names)
        else:
            series = pd.Series(importance)

        # Return only non-zero coefficients
        return series[series > 0].sort_values(ascending=False)

    def get_selected_features(self) -> list[str]:
        """Get features selected by Lasso (non-zero coefficients).

        Returns:
            List of selected feature names
        """
        if not self.is_fitted:
            return []

        nonzero_mask = self.model.coef_ != 0
        if self.feature_names:
            return [f for f, mask in zip(self.feature_names, nonzero_mask) if mask]
        return list(np.where(nonzero_mask)[0])


class ElasticNetVolModel(BaseVolModel):
    """ElasticNet model combining L1 and L2 regularization."""

    name = "elasticnet"

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        max_iter: int = 10000,
        **kwargs: Any,
    ):
        """Initialize the ElasticNet model.

        Args:
            alpha: Regularization strength
            l1_ratio: Balance between L1 and L2 (0 = Ridge, 1 = Lasso)
            fit_intercept: Whether to fit intercept
            normalize_features: Whether to standardize features
            max_iter: Maximum iterations
            **kwargs: Additional parameters
        """
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            normalize_features=normalize_features,
            max_iter=max_iter,
            **kwargs,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.max_iter = max_iter

        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        self.scaler = StandardScaler() if normalize_features else None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "ElasticNetVolModel":
        """Fit the ElasticNet model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

        n_nonzero = np.sum(self.model.coef_ != 0)
        logger.info(
            f"Fitted {self.name} with alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
            f"{n_nonzero} non-zero coefficients"
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)

        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        return self.model.predict(X_arr)

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature coefficients as importance.

        Returns:
            Series with absolute coefficients
        """
        if not self.is_fitted:
            return None

        importance = np.abs(self.model.coef_)
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)
