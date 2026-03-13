# Autoresearch Progress — mar9c

## Key Findings

### What worked (kept)
1. **Higher leverage via gross_max** (1→5x): Massive return improvement
2. **Volatility targeting** (vol_target=0.40): Dynamically scales portfolio to target annualized vol
3. **Uncapped pair weights** (max_pair_w=1.0): Let strategy express full conviction
4. **Vol scalar cap increase** (10x→20x): Biggest single improvement, CAGR from 98% to 208%
5. **Higher individual position limit** (w_max=0.10): Concentration in best names
6. **Weight smoothing** (EWM span=4): Reduces turnover/costs

### What didn't work (discarded)
- **Lower entry_z** (1.0, 1.25): Too much noise, CAGR drops to 10-24%
- **Higher entry_z** (2.0): Too few trades, negative returns
- **Looser correlation** (0.6): Bad pair quality, CAGR 3.2%
- **Tighter correlation** (0.8): Too few pairs, CAGR 19%
- **Shorter train window** (189 days): Much worse, CAGR ~0%
- **EWM z-score**: Negative returns
- **No beta neutralization**: Essential, CAGR drops to 13%
- **Cross-sector pairs**: Noisier, CAGR 68%
- **Momentum confirmation entry**: Delays entries, negative returns
- **Market regime filter**: Cuts profitable periods too
- **Dual-lookback z-score**: Averages with long lookback hurts
- **Vol scalar cap 30x**: Excessive drawdown (-31.7%)

### Key insights
- Vol-targeting is the core return driver — it provides dynamic leverage
- The 20x vol scalar cap is critical: it allows high leverage during calm periods
- Pair quality filters are well-calibrated (corr=0.7, pval=0.03, HL 3-60)
- Beta neutralization is essential for the strategy
- Config is near-optimal: entry_z=1.5, exit_z=0.5, step_window=21, train_window=252
- Same-sector constraint important for pair quality

## Experiments Run: 115-156 (42 experiments)
- Kept: 5 (exp138-142)
- Discarded: 37
