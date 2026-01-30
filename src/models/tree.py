"""Tree-based models for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseVolModel

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class XGBoostVolModel(BaseVolModel):
    """XGBoost model for volatility prediction."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        """Initialize the XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Row sampling ratio
            colsample_bytree: Column sampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_weight: Minimum sum of instance weight in a child
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional XGBoost parameters
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="reg:squarederror",
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: list[tuple] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "XGBoostVolModel":
        """Fit the XGBoost model.

        Args:
            X: Feature matrix
            y: Target values
            eval_set: Evaluation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Whether to print progress
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        fit_params: dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X_arr, y_arr, **fit_params)
        self.is_fitted = True

        logger.info(
            f"Fitted {self.name} with {self.params['n_estimators']} estimators, "
            f"max_depth={self.params['max_depth']}"
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
        return self.model.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series | None:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')

        Returns:
            Series with feature importance
        """
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_

        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)


class LightGBMVolModel(BaseVolModel):
    """LightGBM model for volatility prediction."""

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_child_samples: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        """Initialize the LightGBM model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for unlimited)
            num_leaves: Maximum number of leaves per tree
            learning_rate: Learning rate
            subsample: Row sampling ratio
            colsample_bytree: Column sampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_samples: Minimum samples in a leaf
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional LightGBM parameters
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: list[tuple] | None = None,
        callbacks: list | None = None,
        **kwargs: Any,
    ) -> "LightGBMVolModel":
        """Fit the LightGBM model.

        Args:
            X: Feature matrix
            y: Target values
            eval_set: Evaluation set for early stopping
            callbacks: LightGBM callbacks
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        if callbacks is not None:
            fit_params["callbacks"] = callbacks

        self.model.fit(X_arr, y_arr, **fit_params)
        self.is_fitted = True

        logger.info(
            f"Fitted {self.name} with {self.params['n_estimators']} estimators, "
            f"num_leaves={self.params['num_leaves']}"
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
        return self.model.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series | None:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('split', 'gain')

        Returns:
            Series with feature importance
        """
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_

        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)


class RandomForestVolModel(BaseVolModel):
    """Random Forest model for volatility prediction."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        """Initialize the Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider for split
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters
        """
        from sklearn.ensemble import RandomForestRegressor

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> "RandomForestVolModel":
        """Fit the Random Forest model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

        logger.info(f"Fitted {self.name} with {self.params['n_estimators']} trees")
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
        return self.model.predict(X_arr)

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance.

        Returns:
            Series with feature importance
        """
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_

        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)
