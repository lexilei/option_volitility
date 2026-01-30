.PHONY: install install-dev test lint format type-check clean data train dashboard help

# Default target
help:
	@echo "Option Volatility Strategy - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (ruff)"
	@echo "  make type-check   - Run type checker (mypy)"
	@echo ""
	@echo "Data & Training:"
	@echo "  make data         - Fetch sample data with IV"
	@echo "  make train        - Train all models"
	@echo "  make train-quick  - Train baseline models only"
	@echo ""
	@echo "Backtest:"
	@echo "  make backtest     - Run backtest with baseline model"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard    - Run Streamlit dashboard"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make clean-data   - Remove all data files"

# Installation (using conda)
install:
	conda install -c conda-forge pandas numpy scipy scikit-learn xgboost lightgbm -y
	pip install streamlit plotly optuna massive python-dotenv loguru pydantic pydantic-settings pyarrow requests joblib pytest pytest-cov

install-dev:
	pip install ruff mypy

# Testing
test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

test-quick:
	python -m pytest tests/ -v -x --tb=short

# Linting and formatting
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

type-check:
	mypy src/

# Data operations
data:
	python scripts/fetch_data.py --symbol SPY --days 365 --fetch-iv

data-full:
	python scripts/fetch_data.py --symbol SPY --days 730 --fetch-iv --fetch-options

# Training
train:
	python scripts/train_models.py --model all

train-quick:
	python scripts/train_models.py --model baseline

train-tune:
	python scripts/train_models.py --model all --tune --n-trials 50

# Backtest
backtest:
	python scripts/run_backtest.py --symbol SPY --model baseline --vrp-threshold 0.02

backtest-xgb:
	python scripts/run_backtest.py --symbol SPY --model xgboost --vrp-threshold 0.02

# Dashboard
dashboard:
	streamlit run dashboard/app.py

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-data:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/models/*
