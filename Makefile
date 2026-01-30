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
	@echo "  make data         - Fetch sample data"
	@echo "  make train        - Train all models"
	@echo "  make train-quick  - Train baseline models only"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard    - Run Streamlit dashboard"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make clean-data   - Remove all data files"

# Installation
install:
	uv sync

install-dev:
	uv sync --all-extras

# Testing
test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-quick:
	uv run pytest tests/ -v -x --tb=short

# Linting and formatting
lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

type-check:
	uv run mypy src/

# Data operations
data:
	uv run python scripts/fetch_data.py --symbol SPY --days 365 --fetch-vix

data-full:
	uv run python scripts/fetch_data.py --symbol SPY --days 730 --fetch-vix --fetch-options

# Training
train:
	uv run python scripts/train_models.py --model all

train-quick:
	uv run python scripts/train_models.py --model baseline

train-tune:
	uv run python scripts/train_models.py --model all --tune --n-trials 50

# Dashboard
dashboard:
	uv run streamlit run dashboard/app.py

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
