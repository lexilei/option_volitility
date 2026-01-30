#!/usr/bin/env python
"""Script to train volatility prediction models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.utils.config import get_settings, MODEL_CONFIGS
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train volatility prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    available_models = [
        "all",
        "baseline",
        "historical_mean",
        "ridge",
        "lasso",
        "elasticnet",
        "xgboost",
        "lightgbm",
        "random_forest",
        "lstm",
        "tft",
        "ensemble",
    ]

    parser.add_argument(
        "--model",
        type=str,
        choices=available_models,
        default="all",
        help="Model to train",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to train on",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/models",
        help="Directory to save trained models",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for tuning",
    )

    parser.add_argument(
        "--train-window",
        type=int,
        default=504,
        help="Training window in days",
    )

    parser.add_argument(
        "--test-window",
        type=int,
        default=63,
        help="Test window in days",
    )

    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def generate_sample_data():
    """Generate sample data for demonstration."""
    import numpy as np
    import pandas as pd

    logger.info("Generating sample data for training...")

    np.random.seed(42)
    n_samples = 756  # 3 years

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_samples, freq="B")

    # Price data
    returns = np.random.normal(0.0005, 0.015, n_samples)
    prices = 100 * np.cumprod(1 + returns)

    # Realized volatility
    rv_21d = pd.Series(returns).rolling(21).std() * np.sqrt(252)

    # Implied volatility (with premium)
    iv = rv_21d + np.random.normal(0.02, 0.01, n_samples)
    iv = np.clip(iv, 0.05, 0.8)

    # Features
    features = pd.DataFrame({
        "rv_5d": pd.Series(returns).rolling(5).std() * np.sqrt(252),
        "rv_10d": pd.Series(returns).rolling(10).std() * np.sqrt(252),
        "rv_21d": rv_21d,
        "rv_63d": pd.Series(returns).rolling(63).std() * np.sqrt(252),
        "iv": iv,
        "vrp": iv - rv_21d,
        "rsi_14": 50 + np.cumsum(np.random.normal(0, 5, n_samples)).clip(-50, 50),
        "atr_pct": np.abs(np.random.normal(0.01, 0.005, n_samples)),
        "return_1d": returns,
        "return_5d": pd.Series(returns).rolling(5).sum(),
        "vix": 15 + np.cumsum(np.random.normal(0, 0.5, n_samples)).clip(-5, 65),
    }, index=dates)

    # Target: future realized volatility
    target = rv_21d.shift(-21)

    # Drop NaN
    valid_mask = ~(features.isna().any(axis=1) | pd.isna(target))
    features = features[valid_mask]
    target = target[valid_mask]

    return features, target


def main() -> int:
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    logger.info(f"Training model(s): {args.model}")

    # Generate or load data
    try:
        from src.data.storage import ParquetStorage

        storage = ParquetStorage(args.data_dir)
        features_df = storage.load_processed(f"features/{args.symbol}")

        if features_df is None:
            logger.warning("No processed features found, generating sample data")
            X, y = generate_sample_data()
        else:
            # Prepare data
            from src.features.pipeline import FeaturePipeline

            pipeline = FeaturePipeline()
            X, y = pipeline.prepare_training_data(features_df)

    except Exception as e:
        logger.warning(f"Could not load data: {e}. Using sample data.")
        X, y = generate_sample_data()

    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

    # Import training modules
    from src.training.trainer import ModelTrainer, TrainingConfig

    # Initialize trainer
    trainer = ModelTrainer(X, y)

    # Determine models to train
    if args.model == "all":
        models_to_train = ["historical_mean", "ridge", "lasso", "xgboost", "lightgbm"]
    elif args.model == "baseline":
        models_to_train = ["historical_mean"]
    else:
        models_to_train = [args.model]

    # Train models
    results = {}

    for model_name in models_to_train:
        if model_name == "ensemble":
            continue  # Train ensemble separately

        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")

        try:
            config = TrainingConfig(
                model_name=model_name,
                model_params=MODEL_CONFIGS.get(model_name, {}),
                use_walk_forward=not args.no_walk_forward,
                train_window_days=args.train_window,
                test_window_days=args.test_window,
                tune_hyperparams=args.tune,
                n_trials=args.n_trials,
                save_model=True,
                save_dir=args.save_dir,
            )

            result = trainer.train(config)
            results[model_name] = result

            logger.info(f"Metrics: {result.aggregate_metrics}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    # Train ensemble if requested
    if args.model in ["all", "ensemble"] and len(results) > 1:
        logger.info(f"\n{'='*50}")
        logger.info("Training ensemble")
        logger.info(f"{'='*50}")

        try:
            ensemble_result = trainer.train_ensemble(
                base_model_names=list(results.keys()),
                save_dir=args.save_dir,
            )
            results["ensemble"] = ensemble_result
        except Exception as e:
            logger.error(f"Failed to train ensemble: {e}")

    # Print comparison
    if results:
        logger.info(f"\n{'='*50}")
        logger.info("Model Comparison")
        logger.info(f"{'='*50}")

        comparison = trainer.compare_models()
        logger.info(f"\n{comparison.to_string()}")

    logger.info("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
