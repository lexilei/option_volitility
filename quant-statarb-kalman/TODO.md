# TODO

## Minimum runnable test (now)
- [x] Add `data/raw/sector_map.csv` with `symbol,sector` for the configured universe.
- [ ] Run `statarb download-data --config configs/universe.yaml` and confirm `data/processed/prices.parquet`.
- [ ] Run `statarb select-pairs --config configs/backtest.yaml --prices data/processed/prices.parquet` and verify `data/processed/pairs.csv` is non-empty.
- [ ] Run `statarb backtest --config configs/backtest.yaml --prices data/processed/prices.parquet` and sanity-check metrics + `data/processed/equity.csv`.

## Remaining work before full backtest
- [x] Implement `statarb/kalman.py` validations (sanity checks, parameterization).
- [x] Implement `statarb/signals.py` usage in a per-pair loop (in backtest/generate-weights).
- [x] Implement `statarb/portfolio.py` end-to-end weight aggregation + neutrality checks (in backtest/generate-weights).
- [x] Implement `statarb/costs.py` in backtest loop.
- [x] Implement `statarb/backtest.py` walk-forward engine + metrics outputs.
- [x] Add tests for pairs selection, kalman regression, and backtest integrity.
- [x] Implement CLI commands: `backtest`, `generate-weights`, `trade-paper`.

## Execution loop (paper)
- [x] Implement Alpaca REST integration in `execution/broker_alpaca.py`.
- [x] Add risk checks in `execution/risk_manager.py` (stub).
- [x] Wire `execution/rebalance.py` into `trade-paper` CLI (dry run).
- [ ] Implement live submit in `trade-paper` using `execution.broker_alpaca` (order submit + error handling).
- [ ] Pull live account equity + current positions from Alpaca before sizing orders.
- [ ] Add order type/time-in-force config and rounding rules (lot size / min notional).
- [ ] Expand risk checks to use live PnL and enforce `max_daily_loss_pct`.

## Data + universe
- [ ] Add non-yfinance price source option (e.g., CSV) or document yfinance-only support.
- [ ] Add alternate sector map source (e.g., parquet/JSON) or document CSV-only support.

## Infra / ops
- [x] Add `scripts/run_daily.sh` for pipeline execution.
- [x] Expand `infrastructure/docker/Dockerfile` to copy full project and run CLI.
- [ ] Add logging + alerting (CLI logs, failure notification).
- [ ] Add scheduler/deployment (cron/systemd or AWS EventBridge/Lambda).

## Tests
- [x] Add tests for pairs selection, kalman regression, and backtest integrity.
- [ ] Add tests for execution sizing + risk checks.

## Docs
- [ ] Flesh out root `README.md` with repo overview and entry points.
