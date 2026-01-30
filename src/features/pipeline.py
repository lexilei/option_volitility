"""Feature engineering pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from .volatility import VolatilityCalculator
from .technical import TechnicalIndicators
from .macro import MacroFeatures

if TYPE_CHECKING:
    from pathlib import Path


class FeaturePipeline:
    """Unified feature engineering pipeline."""

    def __init__(
        self,
        rv_windows: list[int] | None = None,
        include_technical: bool = True,
        include_macro: bool = True,
        include_lags: bool = True,
        lag_periods: list[int] | None = None,
    ):
        """Initialize the feature pipeline.

        Args:
            rv_windows: Windows for realized volatility calculation
            include_technical: Whether to include technical indicators
            include_macro: Whether to include macro features
            include_lags: Whether to include lagged features
            lag_periods: Periods for lagged features
        """
        self.rv_windows = rv_windows or [5, 10, 21, 63]
        self.include_technical = include_technical
        self.include_macro = include_macro
        self.include_lags = include_lags
        self.lag_periods = lag_periods or [1, 2, 3, 5]

        self.vol_calc = VolatilityCalculator()
        self.tech_calc = TechnicalIndicators()
        self.macro_calc = MacroFeatures()

        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit the pipeline (store feature names).

        Args:
            df: Training data

        Returns:
            Self
        """
        # Transform to get feature names
        transformed = self.transform(df)
        self.feature_names = list(transformed.columns)
        return self

    def transform(
        self,
        price_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
        iv_data: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Transform price data into features.

        Args:
            price_df: DataFrame with OHLCV columns
            vix_df: DataFrame with VIX data
            market_df: DataFrame with market data (e.g., SPY)
            iv_data: Implied volatility series

        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature engineering pipeline")

        # Start with volatility features
        result = self.vol_calc.compute_all_rv_measures(price_df, self.rv_windows)

        # Add IV-based features if available
        if iv_data is not None:
            result["iv"] = iv_data

            # Volatility risk premium
            for window in self.rv_windows:
                rv_col = f"rv_cc_{window}d"
                if rv_col in result.columns:
                    result[f"vrp_{window}d"] = self.vol_calc.volatility_risk_premium(
                        iv_data, result[rv_col]
                    )
                    result[f"vol_ratio_{window}d"] = self.vol_calc.volatility_ratio(
                        iv_data, result[rv_col]
                    )

            # IV percentile and z-score
            result["iv_percentile"] = self.vol_calc.volatility_percentile(iv_data)
            result["iv_zscore"] = self.vol_calc.volatility_z_score(iv_data)

        # Add RV percentile and z-score for main RV measure
        main_rv = result.get("rv_cc_21d")
        if main_rv is not None:
            result["rv_percentile"] = self.vol_calc.volatility_percentile(main_rv)
            result["rv_zscore"] = self.vol_calc.volatility_z_score(main_rv)

        # Technical indicators
        if self.include_technical:
            tech_features = self.tech_calc.compute_all(price_df)
            # Add only new columns
            new_cols = [c for c in tech_features.columns if c not in result.columns]
            result = pd.concat([result, tech_features[new_cols]], axis=1)

        # Macro features
        if self.include_macro:
            macro_features = self.macro_calc.compute_all(price_df, vix_df, market_df)
            # Add only new columns
            new_cols = [c for c in macro_features.columns if c not in result.columns]
            result = pd.concat([result, macro_features[new_cols]], axis=1)

        # Lagged features
        if self.include_lags:
            result = self._add_lag_features(result)

        # Drop original OHLCV columns to keep only features
        cols_to_drop = ["open", "high", "low", "close", "volume", "vwap", "num_trades", "timestamp"]
        cols_to_drop = [c for c in cols_to_drop if c in result.columns]
        result = result.drop(columns=cols_to_drop)

        logger.info(f"Generated {len(result.columns)} features")
        return result

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for key columns.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with lagged features
        """
        # Key features to lag
        lag_cols = [
            "rv_cc_21d",
            "iv",
            "vrp_21d",
            "rsi_14",
            "atr_pct_14",
            "vix",
            "return_1d",
        ]

        result = df.copy()

        for col in lag_cols:
            if col not in result.columns:
                continue

            for lag in self.lag_periods:
                result[f"{col}_lag{lag}"] = result[col].shift(lag)

        return result

    def fit_transform(
        self,
        price_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
        iv_data: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            price_df: DataFrame with OHLCV columns
            vix_df: DataFrame with VIX data
            market_df: DataFrame with market data
            iv_data: Implied volatility series

        Returns:
            DataFrame with all features
        """
        result = self.transform(price_df, vix_df, market_df, iv_data)
        self.feature_names = list(result.columns)
        return result

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names
        """
        return self.feature_names

    def prepare_training_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "rv_cc_21d",
        forecast_horizon: int = 21,
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training.

        Creates the target variable (future volatility) and aligns features.

        Args:
            features_df: DataFrame with features
            target_col: Column to use as base for target
            forecast_horizon: Days ahead to predict
            dropna: Whether to drop NaN rows

        Returns:
            Tuple of (X features, y target)
        """
        # Target is future volatility
        if target_col not in features_df.columns:
            raise ValueError(f"Target column {target_col} not found in features")

        y = features_df[target_col].shift(-forecast_horizon)
        y.name = f"target_{target_col}_{forecast_horizon}d"

        # Features are current values (exclude the target column itself)
        X = features_df.drop(columns=[target_col], errors="ignore")

        if dropna:
            # Find rows where both X and y are valid
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        return X, y

    def create_sequences(
        self,
        features_df: pd.DataFrame,
        target_col: str = "rv_cc_21d",
        sequence_length: int = 30,
        forecast_horizon: int = 21,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/transformer models.

        Args:
            features_df: DataFrame with features
            target_col: Target column
            sequence_length: Number of timesteps in input sequence
            forecast_horizon: Days ahead to predict

        Returns:
            Tuple of (X sequences, y targets)
        """
        # Prepare aligned data
        X_df, y_series = self.prepare_training_data(
            features_df, target_col, forecast_horizon, dropna=True
        )

        X_values = X_df.values
        y_values = y_series.values

        sequences = []
        targets = []

        for i in range(sequence_length, len(X_values)):
            sequences.append(X_values[i - sequence_length : i])
            targets.append(y_values[i])

        X = np.array(sequences)
        y = np.array(targets)

        logger.info(f"Created {len(X)} sequences of shape {X.shape[1:]}")
        return X, y

    def save_features(self, features_df: pd.DataFrame, path: str | Path) -> None:
        """Save computed features to parquet.

        Args:
            features_df: DataFrame with features
            path: Output path
        """
        features_df.to_parquet(path)
        logger.info(f"Saved features to {path}")

    @staticmethod
    def load_features(path: str | Path) -> pd.DataFrame:
        """Load features from parquet.

        Args:
            path: Input path

        Returns:
            DataFrame with features
        """
        return pd.read_parquet(path)
