"""Walk-forward cross-validation for time series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.models.base import BaseVolModel


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward cross-validation."""

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: dict[str, float]
    model: Any | None = None


class WalkForwardCV:
    """Walk-forward cross-validation for time series models."""

    def __init__(
        self,
        train_window_days: int = 504,
        test_window_days: int = 63,
        step_days: int = 63,
        min_train_samples: int = 252,
        gap_days: int = 0,
    ):
        """Initialize walk-forward CV.

        Args:
            train_window_days: Number of days in training window
            test_window_days: Number of days in test window
            step_days: Number of days to step forward between folds
            min_train_samples: Minimum samples required in training set
            gap_days: Gap between train and test to prevent look-ahead
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.gap_days = gap_days

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        dates: pd.DatetimeIndex | pd.Index | None = None,
    ) -> Generator[WalkForwardFold, None, None]:
        """Generate walk-forward folds.

        Args:
            X: Feature matrix with datetime index or separate dates
            dates: Date index if X doesn't have one

        Yields:
            WalkForwardFold objects
        """
        if isinstance(X, pd.DataFrame) and dates is None:
            dates = pd.DatetimeIndex(X.index)
        elif dates is None:
            raise ValueError("Must provide dates if X is numpy array")

        dates = pd.DatetimeIndex(dates)
        n_samples = len(X)

        fold_id = 0
        start_idx = 0

        while True:
            # Calculate indices
            train_end_idx = start_idx + self.train_window_days
            if train_end_idx >= n_samples:
                break

            test_start_idx = train_end_idx + self.gap_days
            test_end_idx = test_start_idx + self.test_window_days

            if test_end_idx > n_samples:
                test_end_idx = n_samples

            if test_start_idx >= n_samples:
                break

            # Get actual indices
            train_indices = np.arange(start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)

            if len(train_indices) < self.min_train_samples:
                start_idx += self.step_days
                continue

            if len(test_indices) == 0:
                break

            # Get dates
            train_start = dates[start_idx].date()
            train_end = dates[train_end_idx - 1].date()
            test_start = dates[test_start_idx].date()
            test_end = dates[test_end_idx - 1].date()

            yield WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                test_indices=test_indices,
            )

            fold_id += 1
            start_idx += self.step_days

    def get_n_splits(
        self,
        X: pd.DataFrame | np.ndarray,
        dates: pd.DatetimeIndex | pd.Index | None = None,
    ) -> int:
        """Get the number of folds.

        Args:
            X: Feature matrix
            dates: Date index

        Returns:
            Number of folds
        """
        return sum(1 for _ in self.split(X, dates))

    def cross_validate(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict[str, Any] | None = None,
        fit_params: dict[str, Any] | None = None,
        return_models: bool = False,
        verbose: bool = True,
    ) -> list[WalkForwardResult]:
        """Run walk-forward cross-validation.

        Args:
            model_class: Model class to instantiate for each fold
            X: Feature matrix with datetime index
            y: Target series with datetime index
            model_params: Parameters to pass to model constructor
            fit_params: Parameters to pass to model.fit()
            return_models: Whether to include trained models in results
            verbose: Whether to print progress

        Returns:
            List of WalkForwardResult objects
        """
        if model_params is None:
            model_params = {}
        if fit_params is None:
            fit_params = {}

        results = []
        dates = pd.DatetimeIndex(X.index)

        for fold in self.split(X, dates):
            if verbose:
                logger.info(
                    f"Fold {fold.fold_id}: train {fold.train_start} to {fold.train_end}, "
                    f"test {fold.test_start} to {fold.test_end}"
                )

            # Get train/test data
            X_train = X.iloc[fold.train_indices]
            y_train = y.iloc[fold.train_indices]
            X_test = X.iloc[fold.test_indices]
            y_test = y.iloc[fold.test_indices]

            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train, **fit_params)

            # Predict
            predictions = model.predict(X_test)

            # Evaluate
            metrics = model.evaluate(X_test, y_test)

            if verbose:
                logger.info(f"  RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

            result = WalkForwardResult(
                fold_id=fold.fold_id,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                predictions=predictions,
                actuals=y_test.values,
                metrics=metrics,
                model=model if return_models else None,
            )
            results.append(result)

        return results

    def get_combined_predictions(
        self,
        results: list[WalkForwardResult],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine predictions from all folds.

        Args:
            results: List of WalkForwardResult objects

        Returns:
            Tuple of (all_predictions, all_actuals)
        """
        all_predictions = np.concatenate([r.predictions for r in results])
        all_actuals = np.concatenate([r.actuals for r in results])
        return all_predictions, all_actuals

    def get_aggregate_metrics(
        self,
        results: list[WalkForwardResult],
    ) -> dict[str, float]:
        """Calculate aggregate metrics across all folds.

        Args:
            results: List of WalkForwardResult objects

        Returns:
            Dictionary of aggregate metrics
        """
        predictions, actuals = self.get_combined_predictions(results)

        # Remove NaN
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        mse = np.mean((actuals - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actuals - predictions))

        # MAPE
        nonzero_mask = actuals != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((actuals[nonzero_mask] - predictions[nonzero_mask]) / actuals[nonzero_mask])) * 100
        else:
            mape = np.nan

        # R2
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Per-fold statistics
        fold_rmses = [r.metrics["rmse"] for r in results]
        fold_r2s = [r.metrics["r2"] for r in results]

        return {
            "overall_mse": mse,
            "overall_rmse": rmse,
            "overall_mae": mae,
            "overall_mape": mape,
            "overall_r2": r2,
            "mean_fold_rmse": np.mean(fold_rmses),
            "std_fold_rmse": np.std(fold_rmses),
            "mean_fold_r2": np.mean(fold_r2s),
            "std_fold_r2": np.std(fold_r2s),
            "n_folds": len(results),
        }

    def results_to_dataframe(
        self,
        results: list[WalkForwardResult],
    ) -> pd.DataFrame:
        """Convert results to a DataFrame.

        Args:
            results: List of WalkForwardResult objects

        Returns:
            DataFrame with fold metrics
        """
        records = []
        for r in results:
            record = {
                "fold_id": r.fold_id,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "n_train": len(r.actuals),
                **r.metrics,
            }
            records.append(record)

        return pd.DataFrame(records)


class ExpandingWindowCV(WalkForwardCV):
    """Expanding window cross-validation.

    Unlike rolling window, the training set grows with each fold.
    """

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        dates: pd.DatetimeIndex | pd.Index | None = None,
    ) -> Generator[WalkForwardFold, None, None]:
        """Generate expanding window folds.

        Args:
            X: Feature matrix
            dates: Date index

        Yields:
            WalkForwardFold objects
        """
        if isinstance(X, pd.DataFrame) and dates is None:
            dates = pd.DatetimeIndex(X.index)
        elif dates is None:
            raise ValueError("Must provide dates if X is numpy array")

        dates = pd.DatetimeIndex(dates)
        n_samples = len(X)

        fold_id = 0
        train_end_idx = self.train_window_days

        while train_end_idx < n_samples:
            test_start_idx = train_end_idx + self.gap_days
            test_end_idx = test_start_idx + self.test_window_days

            if test_end_idx > n_samples:
                test_end_idx = n_samples

            if test_start_idx >= n_samples:
                break

            # Training always starts from beginning (expanding)
            train_indices = np.arange(0, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)

            if len(train_indices) < self.min_train_samples:
                train_end_idx += self.step_days
                continue

            if len(test_indices) == 0:
                break

            # Get dates
            train_start = dates[0].date()
            train_end = dates[train_end_idx - 1].date()
            test_start = dates[test_start_idx].date()
            test_end = dates[test_end_idx - 1].date()

            yield WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                test_indices=test_indices,
            )

            fold_id += 1
            train_end_idx += self.step_days
