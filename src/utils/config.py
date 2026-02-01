"""Configuration management using Pydantic."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    massive_api_key: str = Field(default="", description="Massive API key (formerly Polygon.io)")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Data directories
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    models_dir: Path = Field(default=Path("data/models"), description="Models directory")

    # Model training
    default_train_window_days: int = Field(
        default=504, description="Default training window (2 years)"
    )
    default_test_window_days: int = Field(
        default=63, description="Default test window (3 months)"
    )
    default_lookback_days: int = Field(
        default=21, description="Default lookback for features"
    )

    # Risk management
    max_position_size: float = Field(
        default=0.1, description="Max position size as fraction of capital"
    )
    max_drawdown_threshold: float = Field(
        default=0.15, description="Max acceptable drawdown"
    )
    stop_loss_multiplier: float = Field(
        default=2.0, description="Stop loss multiplier"
    )

    def ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        for dir_path in [self.data_dir, self.raw_data_dir, self.processed_data_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()


# Model-specific configurations
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "historical_mean": {
        "window": 21,
    },
    "ridge": {
        "alpha": 1.0,
        "normalize_features": True,
    },
    "lasso": {
        "alpha": 0.1,
        "normalize_features": True,
    },
    "elasticnet": {
        "alpha": 0.1,
        "l1_ratio": 0.5,
        "normalize_features": True,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lstm": {
        "sequence_length": 30,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
    },
    "tft": {
        "sequence_length": 30,
        "hidden_size": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
    },
}

# Feature configuration
FEATURE_CONFIG = {
    "rv_windows": [5, 10, 21, 63],
    "technical_indicators": [
        "rsi",
        "atr",
        "bollinger",
        "macd",
        "stochastic",
    ],
    "lag_periods": [1, 2, 3, 5],
}

# Strategy configuration
STRATEGY_CONFIG = {
    "vrp_threshold": 0.02,
    "position_holding_days": 21,
    "max_positions": 3,
    "initial_capital": 100000,
}
