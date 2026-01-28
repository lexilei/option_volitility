# Quant StatArb Kalman

**Goal:** a clean, end-to-end statistical arbitrage system for pairs/spread trading with **dynamic hedge ratios (Kalman filter)**, **sector/beta neutrality**, and **walk‑forward backtesting**. It is built to be understandable by a CS background while still grounded in trading theory.

---

**Contents**
1. What this project is
2. System flow
3. Theoretical foundations (with equations)
4. Implementation map (where each idea lives)
5. Quick start
6. Current results and caveats

---

## 1. What this project is

This repo is a daily (EOD) pairs trading system:
- **Research:** select cointegrated pairs inside the same sector.
- **Model:** estimate a *time‑varying* hedge ratio (beta) via a Kalman filter.
- **Signals:** trade the mean‑reverting spread using z‑score thresholds.
- **Portfolio:** enforce sector and market‑beta neutrality and position limits.
- **Backtest:** walk‑forward evaluation with trading cost modeling.
- **Execution:** paper‑trading stubs (Alpaca) with a dry‑run order generator.

The CLI (`statarb`) wires everything together.

---

## 2. System flow (high level)

1. **Download prices** (yfinance)  
2. **Align calendars + filter symbols**  
3. **Select pairs** (correlation prefilter + Engle‑Granger cointegration)  
4. **Estimate dynamic hedge ratio** (Kalman filter)  
5. **Generate z‑score signals**  
6. **Build weights**  
7. **Neutralize sector & beta**  
8. **Apply limits + costs**  
9. **Walk‑forward backtest**

---

## 3. Theoretical foundations (with equations)

### 3.1 Spread and z‑score

We trade a *spread* between two stocks:

```
s_t = y_t − (α_t + β_t x_t)
```

We then normalize the spread with a rolling z‑score:

```
z_t = (s_t − mean_L(s)) / std_L(s)
```

**Trade rule:**  
Enter when |z| ≥ entry, exit when |z| ≤ exit, stop when |z| ≥ stop.

### 3.2 Kalman filter (dynamic hedge ratio)

We assume β changes over time. A Kalman filter is a recursive estimator that updates:

```
state = [α_t, β_t]
prediction → correction (using observed y_t)
```

This yields a **smooth, adaptive hedge ratio** rather than a single static β.

### 3.3 Beta‑neutral (market neutral)

The portfolio’s market beta is:

```
β_port = Σ w_i β_i
```

We shift weights to target β≈0:

```
w ← w − (β_port − β_target) * β / (Σ β^2 + ε)
```

This removes broad market directionality.

### 3.4 Sector‑neutral

Within each sector, weights are de‑meaned:

```
for sector S:
  w_i ← w_i − mean(w in S)
```

So each sector’s net exposure is ~0.

---

## 4. Implementation map (where each idea lives)

**Data + alignment**
- `statarb/data.py`: price download + calendar alignment

**Pair selection**
- `statarb/pairs.py`
  - Correlation prefilter
  - Engle‑Granger cointegration test
  - Half‑life filter
  - Same‑sector constraint

**Kalman filter**
- `statarb/kalman.py`: dynamic α/β estimation

**Signals**
- `statarb/signals.py`: z‑score + position state machine

**Portfolio + neutrality**
- `statarb/portfolio.py`: weight aggregation, sector neutral, beta neutral, limits

**Backtest**
- `statarb/backtest.py`: walk‑forward engine + turnover cost

**Execution (paper)**
- `execution/rebalance.py`: weights → order deltas
- `execution/risk_manager.py`: gross/net risk checks (stub)
- `execution/broker_alpaca.py`: Alpaca REST wrapper

**Dashboards**
- `statarb/dashboard.py`: data coverage + PnL/positions plots

---

## 5. Quick start

Install:
```bash
python -m pip install -e .
```

Download data + run backtest:
```bash
statarb download-data --config configs/universe.yaml --output data/processed/prices.parquet
statarb select-pairs --config configs/backtest.yaml --prices data/processed/prices.parquet
statarb backtest --config configs/backtest.yaml --prices data/processed/prices.parquet
```

Dashboards:
```bash
statarb dashboard-data --prices data/processed/prices.parquet --output data/processed/dashboard/data
statarb dashboard-pnl --equity data/processed/equity.csv --weights data/processed/weights.parquet --output data/processed/dashboard/pnl
```

---

## 6. Current results and caveats

**Current findings (Jan 26, 2026 run):**
- Universe configured: 200 tickers; retained after missing filter: 194
- Date range: 2018‑01‑02 to 2024‑12‑30 (daily)
- Selected pairs: 1 (PNC‑TFC), p‑value 0.0653
- Backtest metrics: CAGR 0.17%, Sharpe 0.20, Sortino 0.21, MaxDD −1.41%

**How to download and run locally**
```bash
# 1) Clone and enter the repo
git clone <your-repo-url>
cd quant-statarb-kalman

# 2) Create and activate a Python environment (example with conda)
conda create -n trading-env python=3.11 -y
conda activate trading-env

# 3) Install dependencies in editable mode
python -m pip install -e .

# 4) Download data + run a full backtest
statarb download-data --config configs/universe.yaml --output data/processed/prices.parquet
statarb select-pairs --config configs/backtest.yaml --prices data/processed/prices.parquet
statarb backtest --config configs/backtest.yaml --prices data/processed/prices.parquet
```

**When the backtest tends to look good vs. bad**
- **Better regimes:** range‑bound or mean‑reverting markets, stable sector relationships, and periods without prolonged single‑factor dominance.
- **Worse regimes:** strong, persistent trends or regime shifts that break historical relationships between paired stocks.

**Recent underperformance (contextual explanation)**
- The last ~2 years have been unfavorable because **large‑cap tech experienced a sustained, one‑directional rally**.  
- That kind of regime can **break pairs relationships** (spreads trend instead of mean‑revert), so a market‑neutral pairs strategy can look weak or flat.
- In short: **this strategy is not well‑suited to the current market regime**, which has been dominated by a strong tech momentum factor.

**Caveats:**
- `align_calendar` forward‑fills missing prices (may hide gaps)
- Cointegration uses Engle‑Granger only (no Johansen)
- Sector map is CSV‑based and incomplete
- Execution is paper‑only and “dry run” (no live submit)

---

## Repo layout
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
