"""Hyperparameter optimization using Optuna."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if TYPE_CHECKING:
    from src.models.base import BaseVolModel
    from src.training.walk_forward import WalkForwardCV


class HyperoptTuner:
    """Hyperparameter tuner using Optuna."""

    # Default search spaces for different model types
    SEARCH_SPACES = {
        "ridge": {
            "alpha": ("log_float", 1e-4, 1e2),
        },
        "lasso": {
            "alpha": ("log_float", 1e-5, 1e1),
        },
        "elasticnet": {
            "alpha": ("log_float", 1e-5, 1e1),
            "l1_ratio": ("float", 0.0, 1.0),
        },
        "xgboost": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("log_float", 1e-3, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("log_float", 1e-4, 10.0),
            "reg_lambda": ("log_float", 1e-4, 10.0),
            "min_child_weight": ("int", 1, 10),
        },
        "lightgbm": {
            "n_estimators": ("int", 50, 500),
            "num_leaves": ("int", 15, 127),
            "max_depth": ("int", -1, 15),
            "learning_rate": ("log_float", 1e-3, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("log_float", 1e-4, 10.0),
            "reg_lambda": ("log_float", 1e-4, 10.0),
            "min_child_samples": ("int", 5, 100),
        },
        "lstm": {
            "hidden_size": ("int", 32, 256),
            "num_layers": ("int", 1, 4),
            "dropout": ("float", 0.0, 0.5),
            "learning_rate": ("log_float", 1e-4, 1e-2),
            "batch_size": ("categorical", [16, 32, 64, 128]),
        },
        "tft": {
            "hidden_size": ("int", 32, 256),
            "num_heads": ("categorical", [2, 4, 8]),
            "num_layers": ("int", 1, 4),
            "dropout": ("float", 0.0, 0.5),
            "learning_rate": ("log_float", 1e-4, 1e-2),
        },
    }

    def __init__(
        self,
        model_class: type,
        model_name: str | None = None,
        search_space: dict[str, tuple] | None = None,
        n_trials: int = 50,
        cv: "WalkForwardCV | None" = None,
        metric: str = "rmse",
        direction: str = "minimize",
        n_jobs: int = 1,
        random_state: int = 42,
    ):
        """Initialize the tuner.

        Args:
            model_class: Model class to tune
            model_name: Name of the model (for default search space)
            search_space: Custom search space (overrides defaults)
            n_trials: Number of optimization trials
            cv: Cross-validation strategy (defaults to simple train/val split)
            metric: Metric to optimize
            direction: 'minimize' or 'maximize'
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is not installed. Run: pip install optuna")

        self.model_class = model_class
        self.model_name = model_name or model_class.name
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.direction = direction
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set search space
        if search_space is not None:
            self.search_space = search_space
        elif self.model_name in self.SEARCH_SPACES:
            self.search_space = self.SEARCH_SPACES[self.model_name]
        else:
            raise ValueError(
                f"No default search space for {self.model_name}. "
                "Please provide a custom search_space."
            )

        self.study: optuna.Study | None = None
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None
        self.trials_df: pd.DataFrame | None = None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters for a trial.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for param_name, space_def in self.search_space.items():
            space_type = space_def[0]

            if space_type == "int":
                params[param_name] = trial.suggest_int(param_name, space_def[1], space_def[2])
            elif space_type == "float":
                params[param_name] = trial.suggest_float(param_name, space_def[1], space_def[2])
            elif space_type == "log_float":
                params[param_name] = trial.suggest_float(
                    param_name, space_def[1], space_def[2], log=True
                )
            elif space_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, space_def[1])
            else:
                raise ValueError(f"Unknown space type: {space_type}")

        return params

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        fixed_params: dict[str, Any] | None,
    ) -> float:
        """Objective function for optimization.

        Args:
            trial: Optuna trial
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            fixed_params: Parameters that should not be optimized

        Returns:
            Objective value (metric)
        """
        # Suggest parameters
        params = self._suggest_params(trial)

        # Add fixed parameters
        if fixed_params:
            params.update(fixed_params)

        try:
            if self.cv is not None:
                # Use walk-forward CV
                results = self.cv.cross_validate(
                    self.model_class,
                    X,
                    y,
                    model_params=params,
                    verbose=False,
                )
                agg_metrics = self.cv.get_aggregate_metrics(results)
                score = agg_metrics[f"overall_{self.metric}"]
            else:
                # Simple train/val split
                if X_val is None or y_val is None:
                    # Use last 20% as validation
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                else:
                    X_train, y_train = X, y

                model = self.model_class(**params)
                model.fit(X_train, y_train)
                metrics = model.evaluate(X_val, y_val)
                score = metrics[self.metric]

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf") if self.direction == "minimize" else float("-inf")

        return score

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        fixed_params: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run hyperparameter tuning.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            fixed_params: Parameters that should not be optimized
            verbose: Whether to print progress

        Returns:
            Best parameters found
        """
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        # Set verbosity
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Optimize
        logger.info(f"Starting hyperparameter tuning for {self.model_name} ({self.n_trials} trials)")

        self.study.optimize(
            lambda trial: self._objective(trial, X, y, X_val, y_val, fixed_params),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose,
        )

        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.trials_df = self.study.trials_dataframe()

        # Add fixed params to best params
        if fixed_params:
            self.best_params.update(fixed_params)

        logger.info(f"Best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def get_best_model(self) -> "BaseVolModel":
        """Get a model instance with the best parameters.

        Returns:
            Model instance with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        return self.model_class(**self.best_params)

    def get_param_importance(self) -> pd.DataFrame:
        """Get parameter importance from the study.

        Returns:
            DataFrame with parameter importance
        """
        if self.study is None:
            raise ValueError("Must run tune() first")

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return pd.DataFrame(
                {"parameter": list(importance.keys()), "importance": list(importance.values())}
            ).sort_values("importance", ascending=False)
        except Exception as e:
            logger.warning(f"Could not calculate param importance: {e}")
            return pd.DataFrame()

    def plot_optimization_history(self) -> Any:
        """Plot optimization history.

        Returns:
            Plotly figure
        """
        if self.study is None:
            raise ValueError("Must run tune() first")

        return optuna.visualization.plot_optimization_history(self.study)

    def plot_param_importances(self) -> Any:
        """Plot parameter importances.

        Returns:
            Plotly figure
        """
        if self.study is None:
            raise ValueError("Must run tune() first")

        return optuna.visualization.plot_param_importances(self.study)

    def plot_parallel_coordinate(self) -> Any:
        """Plot parallel coordinate visualization.

        Returns:
            Plotly figure
        """
        if self.study is None:
            raise ValueError("Must run tune() first")

        return optuna.visualization.plot_parallel_coordinate(self.study)


def tune_model(
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    n_trials: int = 50,
    metric: str = "rmse",
    cv: "WalkForwardCV | None" = None,
    search_space: dict[str, tuple] | None = None,
    fixed_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], float]:
    """Convenience function to tune a model.

    Args:
        model_class: Model class to tune
        X: Training features
        y: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of trials
        metric: Metric to optimize
        cv: Cross-validation strategy
        search_space: Custom search space
        fixed_params: Fixed parameters
        verbose: Whether to print progress

    Returns:
        Tuple of (best_params, best_score)
    """
    tuner = HyperoptTuner(
        model_class=model_class,
        n_trials=n_trials,
        cv=cv,
        metric=metric,
        search_space=search_space,
    )

    best_params = tuner.tune(X, y, X_val, y_val, fixed_params, verbose)
    return best_params, tuner.best_score
