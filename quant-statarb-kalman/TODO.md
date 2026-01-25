# TODO

## Minimum runnable test (now)
- [x] Add `data/raw/sector_map.csv` with `symbol,sector` for the configured universe.
- [ ] Ensure yfinance download works: `statarb download-data`.
- [ ] Run `statarb select-pairs` and verify `data/processed/pairs.csv` output.

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
- [x] Add risk checks in `execution/risk_manager.py`.
- [x] Wire `execution/rebalance.py` into `trade-paper` CLI.

## Infra / ops
- [x] Add `scripts/run_daily.sh` for pipeline execution.
- [x] Expand `infrastructure/docker/Dockerfile` to copy full project and run CLI.
