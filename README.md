# Option Volatility Strategy with ML Enhancement

A comprehensive framework for volatility selling strategies enhanced with machine learning models. This project combines traditional quantitative finance approaches with modern ML techniques to predict volatility and generate alpha.

## Overview

This project implements a complete pipeline for:
- **Data Collection**: Fetching options and underlying data from Polygon.io
- **Feature Engineering**: Computing volatility metrics (RV, IV, VRP) and technical indicators
- **ML Models**: From simple baselines to advanced deep learning (LSTM, Temporal Fusion Transformer)
- **Backtesting**: Robust walk-forward validation and strategy simulation
- **Visualization**: Interactive Streamlit dashboard for analysis

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd option_volatility

# Install dependencies with uv
uv sync

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your Polygon.io API key
```

### Usage

#### 1. Fetch Data

```bash
# Fetch historical data for SPY
uv run python scripts/fetch_data.py --symbol SPY --days 365
```

#### 2. Train Models

```bash
# Train all models
uv run python scripts/train_models.py --model all

# Or train specific model
uv run python scripts/train_models.py --model xgboost
```

#### 3. Launch Dashboard

```bash
uv run streamlit run dashboard/app.py
```

## Project Structure

```
option_volatility/
├── src/                    # Source code
│   ├── data/              # Data fetching and storage
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── training/          # Training pipeline
│   ├── backtest/          # Backtesting framework
│   └── utils/             # Utilities
├── dashboard/             # Streamlit dashboard
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage (gitignored)
├── tests/                 # Unit tests
└── scripts/               # CLI scripts
```

## Models

| Model | Description | Use Case |
|-------|-------------|----------|
| Historical Mean | Rolling average of past volatility | Baseline |
| Ridge/Lasso | Linear regression with regularization | Interpretable predictions |
| XGBoost | Gradient boosted trees | Strong performance on tabular data |
| LightGBM | Fast gradient boosting | Large datasets |
| LSTM | Long Short-Term Memory network | Capturing temporal patterns |
| TFT | Temporal Fusion Transformer | State-of-the-art time series |
| Ensemble | Weighted combination of models | Robust predictions |

## Features

### Volatility Metrics
- **Realized Volatility (RV)**: Historical price volatility
- **Implied Volatility (IV)**: Market-implied volatility from options
- **Volatility Risk Premium (VRP)**: IV - RV spread

### Technical Indicators
- RSI, ATR, Bollinger Bands
- Moving averages (SMA, EMA)
- Volume metrics

### Macro Features
- VIX levels and term structure
- Interest rate spreads
- Market regime indicators

## Dashboard Pages

1. **Data Explorer**: Visualize raw data and basic statistics
2. **Feature Analysis**: Analyze feature distributions and correlations
3. **Model Comparison**: Compare model performance metrics
4. **Backtest Results**: View strategy performance and risk metrics
5. **Settings**: Configure parameters and API keys

## Configuration

### Environment Variables

Create a `.env` file with:

```env
POLYGON_API_KEY=your_api_key_here
LOG_LEVEL=INFO
DATA_DIR=data
```

### Model Configuration

Models can be configured via `src/utils/config.py` or the dashboard settings page.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy src/
```

## Key Concepts

### Walk-Forward Validation

To avoid look-ahead bias, we use walk-forward cross-validation:
- Training window: 2 years rolling
- Test window: 3 months
- Re-train models at each step

### Volatility Risk Premium (VRP)

The core strategy exploits the tendency of implied volatility to exceed realized volatility:

```
VRP = IV - RV
```

When VRP is high, selling volatility (e.g., selling options) tends to be profitable.

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Annual return / Max drawdown

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Polygon.io](https://polygon.io/) for market data
- [Streamlit](https://streamlit.io/) for the dashboard framework
- The quantitative finance community for research and inspiration
