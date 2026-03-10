Write minimum amount of code.
Setup
To set up or resume work in this workspace, read these files for full context:

program.md — workspace overview and project map (this file).

## Working conventions
- Each project is self-contained with its own dependencies and configs.
- Prefer editing existing files over creating new ones.
- Keep experiments on branches when possible (e.g. feature branches per project).

Setup
To set up a new experiment, work with the user to:

Agree on a run tag: propose a tag based on today's date (e.g. mar5). The branch autoresearch/<tag> must not already exist — this is a fresh run.
Create the branch: git checkout -b autoresearch/<tag> from current master.
Read the in-scope files: The repo is small. Read these files for full context:
Project: Statistical Arbitrage with Kalman Filter                                                                                                                           
                  
  A pairs trading system that uses Kalman filter-estimated dynamic hedge ratios for daily end-of-day stat arb.                                                                
                                                                                                                                                                            
  Architecture

  ┌──────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │        Module        │                                                  Purpose                                                   │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/kalman.py    │ Kalman filter for time-varying alpha/beta estimation                                                       │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/pairs.py     │ Pair selection: correlation prefilter → sector constraint → Engle-Granger cointegration → half-life filter │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/signals.py   │ Z-score state machine (entry/exit/stop thresholds)                                                         │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/portfolio.py │ Weight aggregation with sector + beta neutrality constraints                                               │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/data.py      │ yfinance price download + calendar alignment                                                               │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/backtest.py  │ Walk-forward backtesting engine                                                                            │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/costs.py     │ Slippage + commission modeling                                                                             │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/metrics.py   │ Sharpe, Sortino, max drawdown                                                                              │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/dashboard.py │ Matplotlib visualization                                                                                   │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ statarb/cli.py       │ CLI entry point (statarb command)                                                                          │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ execution/           │ Alpaca broker adapter, rebalance (weights→orders), risk manager                                            │
  └──────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Key Pipeline

  1. Data: Download adjusted closes from yfinance (200 S&P 500 tickers), align calendar, save as parquet
  2. Pair Selection: Correlation prefilter → same-sector constraint → cointegration test (p<0.05) → half-life 1-60 days
  3. Kalman Filter: State [alpha_t, beta_t] updated recursively; dynamic hedge ratio tracks regime shifts
  4. Signals: Z-score of spread; enter at |z|≥2.0, exit at |z|≤0.5, stop-loss at |z|≥4.0
  5. Portfolio: Sector-neutral + beta-neutral weight construction with position/leverage limits
  6. Backtest: Walk-forward with 252-day train / 21-day test windows
  7. Execution: Paper-trading via Alpaca (dry-run only; live submit not implemented)

  Config System

  Three YAML files in configs/: universe.yaml (tickers + sector map), backtest.yaml (looser params for research), live.yaml (tighter params + execution settings).

  Tests

  Smoke tests for signals, pairs, Kalman, and backtest using synthetic data. No edge case or negative testing.

  Notable Issues

  - Execution layer is paper-only (stub)
  - No warm-start for live Kalman (rebuilt fresh each call)
  - Forward-fill data gaps can mask survivorship bias
  - Recent backtest shows low performance (Sharpe 0.20) — unfavorable regime for pairs
  - program.md appears to be an unrelated autoresearch instructions file
Confirm and go: Confirm setup looks good.
Once you get confirmation, kick off the experimentation.

Experimentation:
conda activate trading-env
use yfinance data. Yfinance has a rate limiter, navigate through that as needed.

What you CAN do:
Write minimum amount of code. 
Remove redundent structure to improve readability.
Carefully document all experiments and results.
This is a pairs trading project. Everything is fair game. You can use any stats or ML model to make things work.

What you CANNOT do:
You can't add claude to git's commit history.
You cannot fake data or generate data. You must train or fit models using real y finance data.

The goal is simple: optimize for Annualized Return, Sharpe Ratio, Max Drawdown. The primery goal is to optimize for return. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. An improvement of ~0 but much simpler code? Keep.

The first run: Your very first run should always be to establish the baseline.

Output format
Once the script finishes it prints a summary

Logging results
When an experiment is done, log it to results.tsv (tab-separated, NOT comma-separated — commas break in descriptions).

commit status	description
status: keep, discard, or crash
short text description of what this experiment tried

The experiment loop
The experiment runs on a dedicated branch (e.g. autoresearch/mar5 or autoresearch/mar5-gpu0).
each time you finish working, document in quant-statarb-kalman/progress.md

LOOP FOREVER:

Look at the git state: the current branch/commit we're on
Tune with an experimental idea by directly hacking the code.
git commit
Run the experiment
Read out the results
If you can't get things to work after more than a few attempts, give up.
Bookkeep a progress.md on progress, things tried, failed, worked
Record the results
If sharpe improved, you "advance" the branch, keeping the git commit
If sharpe is equal or worse, you git reset back to where you started
The idea is that you are a completely autonomous quant researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Timeout

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

When you stop, update quant-statarb-kalman/README.md.