#!/usr/bin/env python
"""Script to fetch market data from Polygon.io."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data import DataFetcher
from src.utils.config import get_settings
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch market data from Polygon.io",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Stock ticker symbol",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --days if provided.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )

    parser.add_argument(
        "--fetch-options",
        action="store_true",
        help="Also fetch options chain data",
    )

    parser.add_argument(
        "--fetch-vix",
        action="store_true",
        help="Also fetch VIX data",
    )

    parser.add_argument(
        "--fetch-iv",
        action="store_true",
        help="Fetch current ATM IV from options chain (requires Options plan)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache and fetch fresh data",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store data",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Massive API key (or set MASSIVE_API_KEY env var)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Get API key
    settings = get_settings()
    api_key = args.api_key or settings.massive_api_key

    if not api_key:
        logger.error(
            "No API key provided. Set MASSIVE_API_KEY environment variable "
            "or use --api-key argument."
        )
        return 1

    # Calculate dates
    end_date = date.today()
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)

    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = end_date - timedelta(days=args.days)

    logger.info(f"Fetching data for {args.symbol} from {start_date} to {end_date}")

    # Initialize fetcher
    fetcher = DataFetcher(
        api_key=api_key,
        data_dir=args.data_dir,
        cache_enabled=not args.no_cache,
    )

    try:
        # Fetch price data
        logger.info("Fetching price history...")
        price_df = fetcher.get_price_history(
            args.symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=not args.no_cache,
        )

        if price_df.empty:
            logger.warning(f"No price data returned for {args.symbol}")
        else:
            logger.info(f"Fetched {len(price_df)} price records")
            logger.info(f"Date range: {price_df.index.min()} to {price_df.index.max()}")

        # Fetch VIX if requested
        if args.fetch_vix:
            logger.info("Fetching VIX history...")
            vix_df = fetcher.get_vix_history(
                start_date=start_date,
                end_date=end_date,
                use_cache=not args.no_cache,
            )

            if vix_df.empty:
                logger.warning("No VIX data returned")
            else:
                logger.info(f"Fetched {len(vix_df)} VIX records")

        # Fetch options if requested
        if args.fetch_options:
            logger.info("Fetching options chain...")
            options_df = fetcher.get_options_chain(
                args.symbol,
                use_cache=not args.no_cache,
            )

            if options_df.empty:
                logger.warning(f"No options data returned for {args.symbol}")
            else:
                logger.info(f"Fetched {len(options_df)} options contracts")

        # Fetch IV from options chain if requested
        if args.fetch_iv:
            logger.info("Fetching ATM IV from options chain...")
            try:
                from src.data.massive_client import MassiveClient
                from src.data.storage import ParquetStorage
                import pandas as pd

                massive_client = MassiveClient(api_key)

                # Get current ATM IV
                iv = massive_client.get_atm_iv(args.symbol, days_to_expiry=30)

                if iv is not None:
                    logger.info(f"Current ATM IV: {iv:.4f} ({iv*100:.2f}%)")

                    # Save IV data point
                    storage = ParquetStorage(args.data_dir)
                    iv_data = pd.DataFrame({
                        "date": [pd.Timestamp.today().date()],
                        "atm_iv": [iv],
                        "symbol": [args.symbol],
                    }).set_index("date")

                    # Append to existing IV history
                    key = f"iv/{args.symbol}"
                    if storage.exists(key):
                        storage.append(iv_data, key, dedupe_column="symbol")
                    else:
                        storage.save(iv_data, key)

                    logger.info(f"Saved IV data to {key}")
                else:
                    logger.warning("Could not calculate ATM IV")
            except ImportError:
                logger.error("massive library not installed. Run: pip install massive")
            except Exception as e:
                logger.error(f"Error fetching IV: {e}")

        logger.info("Data fetch complete!")
        return 0

    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
