#!/usr/bin/env python
"""Daily paper trading script.

Run this daily to:
1. Fetch latest market data and IV
2. Generate trading signals
3. Execute paper trades
4. Update portfolio
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.utils.logging import setup_logging
from src.utils.config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily paper trading update",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to trade",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital (only used on first run)",
    )

    parser.add_argument(
        "--position-size",
        type=float,
        default=10000,
        help="Position size per trade",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions",
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
        "--reset",
        action="store_true",
        help="Reset paper trading state",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print performance summary only",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def fetch_latest_data(symbol: str, api_key: str, use_cache: bool = True) -> dict | None:
    """Fetch latest price and IV data.

    Returns:
        Dict with 'price', 'iv', 'rv' or None if failed
    """
    from src.data.storage import ParquetStorage
    from src.features.volatility import VolatilityCalculator
    import pandas as pd
    import time

    storage = ParquetStorage("data")
    vol_calc = VolatilityCalculator()

    # Try to use cached data first
    if use_cache:
        try:
            price_df = storage.load(f"prices/{symbol}")
            iv_df = storage.load(f"iv/{symbol}")

            if price_df is not None and not price_df.empty:
                latest_price = price_df["close"].iloc[-1]
                rv = vol_calc.realized_volatility(price_df["close"], window=21)
                latest_rv = rv.iloc[-1] if not rv.empty and not pd.isna(rv.iloc[-1]) else 0.12

                if iv_df is not None and not iv_df.empty:
                    iv = iv_df["atm_iv"].iloc[-1]
                else:
                    iv = latest_rv + 0.08  # Default VRP of 8%

                logger.info(f"Using cached data for {symbol}")
                logger.info(f"Price: ${latest_price:.2f}, IV: {iv:.4f}, RV: {latest_rv:.4f}")

                return {
                    "price": latest_price,
                    "iv": iv,
                    "rv": latest_rv,
                    "vrp": iv - latest_rv,
                }
        except Exception as e:
            logger.warning(f"Could not load cached data: {e}")

    # Try to fetch fresh data
    try:
        from src.data.massive_client import MassiveClient

        client = MassiveClient(api_key)

        # Get latest price with retry
        logger.info(f"Fetching latest data for {symbol}...")

        for attempt in range(3):
            try:
                price_df = client.get_aggregates(symbol, limit=30)
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Rate limited, waiting 5s... (attempt {attempt + 1})")
                    time.sleep(5)
                else:
                    raise

        if price_df.empty:
            logger.error("Failed to fetch price data")
            return None

        latest_price = price_df["close"].iloc[-1]
        logger.info(f"Latest price: ${latest_price:.2f}")

        # Calculate RV
        rv = vol_calc.realized_volatility(price_df["close"], window=21)
        latest_rv = rv.iloc[-1] if not rv.empty else 0.15

        # Get IV from options chain (with delay to avoid rate limit)
        time.sleep(2)

        try:
            iv = client.get_atm_iv(symbol, days_to_expiry=30)
        except Exception:
            iv = None

        if iv is None:
            logger.warning("Could not fetch IV, using estimated value")
            iv = latest_rv + 0.08  # Estimate IV as RV + 8%

        logger.info(f"IV: {iv:.4f} ({iv*100:.2f}%)")
        logger.info(f"RV: {latest_rv:.4f} ({latest_rv*100:.2f}%)")
        logger.info(f"VRP: {(iv-latest_rv):.4f} ({(iv-latest_rv)*100:.2f}%)")

        # Save latest IV
        iv_data = pd.DataFrame({
            "date": [pd.Timestamp.today().date()],
            "atm_iv": [iv],
            "rv_21d": [latest_rv],
            "vrp": [iv - latest_rv],
            "price": [latest_price],
            "symbol": [symbol],
        }).set_index("date")

        key = f"iv/{symbol}"
        if storage.exists(key):
            storage.append(iv_data, key, dedupe_column="symbol")
        else:
            storage.save(iv_data, key)

        return {
            "price": latest_price,
            "iv": iv,
            "rv": latest_rv,
            "vrp": iv - latest_rv,
        }

    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        return None


def main() -> int:
    """Main function."""
    args = parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Get API key
    settings = get_settings()
    api_key = settings.polygon_api_key

    if not api_key:
        logger.error("No API key found. Set POLYGON_API_KEY environment variable.")
        return 1

    # Initialize paper trader
    from src.trading.paper_trader import PaperTrader

    trader = PaperTrader(
        data_dir=args.data_dir,
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        max_positions=args.max_positions,
        holding_days=args.holding_days,
        vrp_threshold=args.vrp_threshold,
    )

    # Handle reset
    if args.reset:
        trader.reset()
        logger.info("Paper trading state has been reset")
        return 0

    # Handle summary only
    if args.summary:
        summary = trader.get_performance_summary()
        if summary:
            print("\n" + "=" * 50)
            print("PAPER TRADING PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"Period: {summary.get('start_date')} to {summary.get('end_date')}")
            print(f"Trading Days: {summary.get('trading_days')}")
            print(f"\nInitial Capital: ${summary.get('initial_capital'):,.2f}")
            print(f"Final Equity: ${summary.get('final_equity'):,.2f}")
            print(f"Total Return: {summary.get('total_return'):.2%}")
            print(f"Annualized Return: {summary.get('annualized_return'):.2%}")
            print(f"Sharpe Ratio: {summary.get('sharpe_ratio'):.2f}")
            print(f"\nTotal Trades: {summary.get('total_trades')}")
            print(f"Win Rate: {summary.get('win_rate'):.2%}")
            print(f"Profit Factor: {summary.get('profit_factor'):.2f}")
            print(f"\nOpen Positions: {summary.get('open_positions')}")
            print(f"Cash: ${summary.get('cash'):,.2f}")
            print("=" * 50)
        else:
            print("No trading history yet.")
        return 0

    # Fetch latest data
    data = fetch_latest_data(args.symbol, api_key)

    if data is None:
        logger.error("Failed to fetch market data")
        return 1

    # Run daily update
    current_date = date.today()
    summary = trader.run_daily_update(
        current_date=current_date,
        iv=data["iv"],
        rv=data["rv"],
        underlying_price=data["price"],
    )

    # Print summary
    print("\n" + "=" * 50)
    print(f"DAILY UPDATE: {summary['date']}")
    print("=" * 50)
    print(f"Market Data:")
    print(f"  IV: {summary['iv']:.4f} ({summary['iv']*100:.2f}%)")
    print(f"  RV: {summary['rv']:.4f} ({summary['rv']*100:.2f}%)")
    print(f"  VRP: {summary['vrp']:.4f} ({summary['vrp']*100:.2f}%)")
    print(f"\nPortfolio:")
    print(f"  Equity: ${summary['equity']:,.2f}")
    print(f"  Cash: ${summary['cash']:,.2f}")
    print(f"  Open Positions: {summary['open_positions']}")
    print(f"\nActivity:")
    print(f"  New Positions: {summary['new_positions']}")
    print(f"  Closed Trades: {summary['closed_trades']}")
    print(f"\nPerformance:")
    print(f"  Daily Return: {summary['daily_return']:.2%}")
    print(f"  Cumulative Return: {summary['cumulative_return']:.2%}")
    print("=" * 50)

    # Output for GitHub Actions
    output_file = Path(args.data_dir) / "paper_trading" / "latest_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
