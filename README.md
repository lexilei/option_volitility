# 📈 Option Volatility Strategy with ML Enhancement

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lexilei-option-volitility.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for **volatility selling strategies** enhanced with machine learning. This project combines traditional quantitative finance with modern ML techniques to predict volatility and capture the **Volatility Risk Premium (VRP)**.

## 🎯 Live Demo

**[👉 View Live Dashboard](https://lexilei-option-volitility.streamlit.app)**

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Real-time Data** | Integration with Massive (Polygon.io) API for price and options data |
| 🔬 **90+ Features** | Volatility metrics, technical indicators, macro factors |
| 🤖 **7 ML Models** | From Historical Mean to Temporal Fusion Transformer |
| 📈 **Backtesting** | Walk-forward validation with comprehensive metrics |
| 💰 **Paper Trading** | Live simulation with automated daily updates |
| 🎨 **Dashboard** | Interactive Streamlit interface for analysis |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- [Massive API Key](https://massive.com) (formerly Polygon.io)

### Installation

```bash
# Clone the repository
git clone https://github.com/lexilei/option_volitility.git
cd option_volitility

# Create conda environment
conda create -n options-env python=3.11 -y
conda activate options-env

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API key
cp .env.example .env
# Edit .env with your API key
```

### Usage

```bash
# Fetch data with IV
python scripts/fetch_data.py --symbol SPY --days 365 --fetch-iv

# Run backtest
python scripts/run_backtest.py --symbol SPY --model baseline

# Start paper trading
python scripts/paper_trade.py --symbol SPY

# Launch dashboard
streamlit run dashboard/app.py
```

## 📊 Strategy Overview

The strategy exploits the **Volatility Risk Premium (VRP)** - the tendency for implied volatility to exceed realized volatility:

```
VRP = IV - RV
```

When VRP > threshold → **Sell Volatility** (profit if IV overestimates RV)
When VRP < -threshold → **Buy Volatility** (profit if IV underestimates RV)

### Signal Flow

```
Market Data → Feature Engineering → ML Prediction → VRP Calculation → Trading Signal
```

## 🤖 ML Models

| Model | Description | Use Case |
|-------|-------------|----------|
| Historical Mean | Rolling average baseline | Benchmark |
| Ridge/Lasso | Linear models with regularization | Fast, interpretable |
| XGBoost | Gradient boosting | High accuracy |
| LightGBM | Fast gradient boosting | Large datasets |
| LSTM | Recurrent neural network | Sequential patterns |
| TFT | Temporal Fusion Transformer | State-of-the-art |
| Ensemble | Combined predictions | Robust performance |

## 📈 Backtesting Results

Example results using SPY data:

| Metric | Value |
|--------|-------|
| Total Return | 23.29% |
| Sharpe Ratio | 1.00 |
| Max Drawdown | 14.24% |
| Win Rate | 90.91% |
| Profit Factor | 2.58 |

## 💰 Paper Trading

Automated daily paper trading with GitHub Actions:

- Runs at 5 PM ET on weekdays (after market close)
- Fetches latest market data and IV
- Generates signals and executes paper trades
- Updates portfolio and performance metrics
- Commits results to repository

```bash
# Check paper trading status
python scripts/paper_trade.py --summary

# Reset paper trading
python scripts/paper_trade.py --reset
```

## 📁 Project Structure

```
option_volitility/
├── src/
│   ├── data/          # Data fetching and storage
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   ├── training/      # Training pipeline
│   ├── backtest/      # Backtesting framework
│   └── trading/       # Paper trading system
├── dashboard/         # Streamlit dashboard
├── scripts/           # CLI scripts
├── tests/             # Unit tests
└── data/              # Data storage
```

## 🛠️ Development

```bash
# Run tests
make test

# Format code
make format

# Type check
make type-check
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Massive](https://massive.com) (formerly Polygon.io) for market data API
- [Streamlit](https://streamlit.io) for the dashboard framework
- [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai), [PyTorch](https://pytorch.org) for ML

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/lexilei">lexilei</a>
</p>
