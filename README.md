# Quant StatArb Kalman

Goal: sector/factor-neutral stat-arb (pairs/spread) with Kalman-filter dynamic hedge ratios, local backtests, and a paper-trading loop (Alpaca), with a path to AWS scheduling.

Constraints:
- Frequency: daily (EOD) first; extendable to hourly.
- Universe: S&P 500 or custom pool (>=200 symbols).
- Neutrality: market beta ~0 and sector-neutral (GICS).
- Costs: commission + slippage (bps) + borrow fee (optional).
- Broker: Alpaca paper (later IBKR).
- Ops: scheduled pipeline + logs + alerts.

Repo layout:
```
quant-statarb-kalman/
├── pyproject.toml
├── README.md
├── configs/
│   ├── universe.yaml
│   ├── backtest.yaml
│   └── live.yaml
├── data/
│   ├── raw/
│   └── processed/
├── statarb/
├── execution/
├── infrastructure/
└── tests/
```

Assumptions:
- Signal is generated on close and traded next day (configurable).
- No lookahead bias: selection and parameters use rolling windows.
- Data download source is pluggable via `statarb.data.fetch_prices`.

Milestones:
- A: data layer + pair selection.
- B: Kalman + signals.
- C: portfolio neutrality + walk-forward backtest.
- D: paper trading loop.
- E: AWS scheduling.

Risky/tuning choices (review before relying on results):
- `prices.max_missing_pct` set to 0.2 to keep more symbols; this can retain stale/partial series.
- `align_calendar` forward-fills missing prices; this may hide gaps and bias cointegration tests.
- `corr_threshold` set to 0.8; pairs below this correlation are excluded before cointegration tests.
- `pval_thresh` set to 0.1 in backtest config; weaker statistical evidence for pairing.
- `sector_map.csv` currently marks most symbols as `UNKNOWN`; sector neutrality isn't enforced yet.

Quick start (after implementation):
- `statarb download-data --config configs/universe.yaml`
- `statarb select-pairs --config configs/backtest.yaml`
- `statarb backtest --config configs/backtest.yaml`
