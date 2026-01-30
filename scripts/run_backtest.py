#!/usr/bin/env python
"""Script to run strategy backtests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run volatility strategy backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to backtest",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Model to use for predictions",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model file",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )

    parser.add_argument(
        "--vrp-threshold",
        type=float,
        default=0.02,
        help="VRP threshold for signals",
    )

    parser.add_argument(
        "--holding-days",
        type=int,
        default=21,
        help="Position holding period",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def generate_sample_data():
    """Generate sample data for backtesting."""
    np.random.seed(42)
    n_days = 504

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    # Generate price data
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n_days),
    }, index=dates)

    return df


def main() -> int:
    """Main function."""
    args = parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    logger.info(f"Running backtest for {args.symbol}")
    logger.info(f"Model: {args.model}")
    logger.info(f"VRP Threshold: {args.vrp_threshold}")

    # Load or generate data
    try:
        from src.data.storage import ParquetStorage

        storage = ParquetStorage(args.data_dir)
        price_df = storage.load(f"prices/{args.symbol}")

        if price_df is None:
            logger.warning("No cached data found, generating sample data")
            price_df = generate_sample_data()
    except Exception as e:
        logger.warning(f"Could not load data: {e}. Using sample data.")
        price_df = generate_sample_data()

    logger.info(f"Data shape: {price_df.shape}")

    # Generate features
    from src.features.pipeline import FeaturePipeline
    from src.features.volatility import VolatilityCalculator

    pipeline = FeaturePipeline()
    vol_calc = VolatilityCalculator()

    features = pipeline.transform(price_df)

    # Calculate IV (simulated for demo)
    rv_21d = vol_calc.realized_volatility(price_df["close"], window=21)
    iv = rv_21d + np.random.normal(0.02, 0.01, len(rv_21d))
    iv = pd.Series(np.clip(iv, 0.05, 0.8), index=price_df.index)

    # Load or create model
    if args.model_path and Path(args.model_path).exists():
        from src.models.base import BaseVolModel

        model = BaseVolModel.load(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.info(f"Training new {args.model} model")

        # Prepare training data
        features["rv_cc_21d"] = rv_21d
        X, y = pipeline.prepare_training_data(features, target_col="rv_cc_21d")

        # Train model
        if args.model == "xgboost":
            from src.models.tree import XGBoostVolModel

            model = XGBoostVolModel(n_estimators=100, max_depth=6)
        elif args.model == "ridge":
            from src.models.linear import RidgeVolModel

            model = RidgeVolModel(alpha=1.0)
        else:
            from src.models.baseline import HistoricalMeanModel

            model = HistoricalMeanModel(window=21)

        model.fit(X, y)

    # Run backtest
    from src.backtest.strategy import VolatilityStrategy
    from src.backtest.metrics import calculate_all_metrics

    strategy = VolatilityStrategy(
        model=model,
        vrp_threshold=args.vrp_threshold,
        position_holding_days=args.holding_days,
    )

    # Generate signals
    logger.info("Generating signals...")
    signals_df = strategy.generate_signals(features, iv, rv_21d)

    # Run backtest
    logger.info("Running backtest...")
    trades_df = strategy.backtest(signals_df, rv_21d)

    if trades_df.empty:
        logger.warning("No trades generated")
        return 0

    # Get equity curve
    equity_df = strategy.get_equity_curve(signals_df, rv_21d, args.initial_capital)

    # Calculate metrics
    if not equity_df.empty:
        metrics = calculate_all_metrics(equity_df["equity"], trades_df)

        logger.info("\n" + "=" * 50)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"\nTotal Return: {metrics.total_return:.2%}")
        logger.info(f"Annualized Return: {metrics.annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        logger.info(f"Win Rate: {metrics.win_rate:.2%}")
        logger.info(f"Total Trades: {metrics.total_trades}")
        logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results = {
                "metrics": metrics.to_dict(),
                "equity": equity_df.to_dict(),
                "trades": trades_df.to_dict(),
            }

            import json

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"\nResults saved to {output_path}")

    logger.info("\nBacktest complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
