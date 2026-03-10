"""Walk-forward backtest."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .costs import apply_costs_from_weights
from .kalman import kalman_regression
from .pairs import Pair, select_pairs
from .portfolio import (
    aggregate_pair_weights,
    apply_limits,
    neutralize_beta,
    neutralize_by_sector,
)
from .signals import compute_zscore, generate_pair_positions


def _estimate_betas(returns: pd.DataFrame, spy_ret: pd.Series) -> pd.Series:
    common = returns.index.intersection(spy_ret.index)
    r = returns.loc[common]
    s = spy_ret.loc[common]
    if s.var(ddof=0) == 0:
        return pd.Series(0.0, index=returns.columns)
    cov = r.apply(lambda x: np.cov(x, s, ddof=0)[0, 1])
    return cov / s.var(ddof=0)


def walk_forward_backtest(
    prices: pd.DataFrame,
    sector_map: Dict[str, str],
    spy: pd.Series,
    train_window: int = 252,
    test_step: int = 21,
    corr_lookback: int = 252,
    corr_threshold: float | None = None,
    max_pairs: int = 50,
    pval_thresh: float = 0.05,
    half_life_min: int = 2,
    half_life_max: int = 20,
    rank_by: str = "pvalue",
    kalman_R: float = 1e-3,
    kalman_Q: float = 1e-4,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    max_pair_w: float = 0.02,
    w_max: float = 0.02,
    gross_max: float = 1.0,
    slippage_bps: float = 2.0,
    proportional: bool = False,
    weight_smooth: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return equity curve, pair positions, and daily weights."""
    # Drop columns with >20% missing before dropping rows
    missing_pct = prices.isna().mean()
    prices = prices.loc[:, missing_pct <= 0.20]
    prices = prices.dropna(how="any")
    returns = prices.pct_change().dropna()
    spy_ret = spy.pct_change().dropna()

    if len(prices) < train_window + test_step:
        raise ValueError("Not enough data for walk-forward backtest")

    all_weights = []
    all_pair_pos = []

    idx = prices.index
    start = train_window
    while start < len(idx):
        train_idx = idx[start - train_window : start]
        test_idx = idx[start : min(start + test_step, len(idx))]
        train_prices = prices.loc[train_idx]

        pairs = select_pairs(
            prices=train_prices,
            sector_map=sector_map,
            train_window=train_window,
            corr_lookback=corr_lookback,
            corr_threshold=corr_threshold,
            min_half_life=half_life_min,
            max_half_life=half_life_max,
            pval_thresh=pval_thresh,
            max_pairs=max_pairs,
            rank_by=rank_by,
        )

        train_ret = returns.reindex(train_idx).dropna(how="any")
        train_spy = spy_ret.reindex(train_idx).dropna()
        common = train_ret.index.intersection(train_spy.index)
        if common.empty:
            step_betas = pd.Series(0.0, index=prices.columns)
        else:
            step_betas = _estimate_betas(train_ret.loc[common], train_spy.loc[common])

        step_weights = pd.DataFrame(index=test_idx)
        step_pair_pos = pd.DataFrame(index=test_idx)

        for pair in pairs:
            window_prices = prices.loc[train_idx.union(test_idx)]
            y = window_prices[pair.y]
            x = window_prices[pair.x]
            _, beta_s, spread = kalman_regression(y, x, R=kalman_R, Q=kalman_Q)
            lookback = max(10, pair.half_life)
            z = compute_zscore(spread, lookback=lookback)
            pos = generate_pair_positions(z, entry=entry_z, exit=exit_z, stop=stop_z, proportional=proportional)
            pos = pos.reindex(window_prices.index).fillna(0.0)

            key = f"{pair.y}-{pair.x}"
            pair_pos = pos.loc[test_idx]
            step_pair_pos[key] = pair_pos

            for date in test_idx:
                signal = float(pair_pos.loc[date])
                beta_t = float(beta_s.loc[date])
                raw = aggregate_pair_weights(
                    pairs=[pair],
                    pair_positions={key: signal},
                    betas_t={key: beta_t},
                    max_pair_w=max_pair_w,
                )
                step_weights = step_weights.reindex(columns=step_weights.columns.union(raw.index))
                row = step_weights.loc[date, raw.index].fillna(0.0)
                step_weights.loc[date, raw.index] = row + raw

        for date in test_idx:
            if date not in step_weights.index:
                continue
            w = step_weights.loc[date].fillna(0.0)
            w = neutralize_beta(w, step_betas, target_beta=0.0)
            w = apply_limits(w, w_max=w_max, gross_max=gross_max)
            step_weights.loc[date, w.index] = w

        all_weights.append(step_weights)
        all_pair_pos.append(step_pair_pos)
        start += test_step

    weights = pd.concat(all_weights).sort_index().fillna(0.0)
    pair_positions = pd.concat(all_pair_pos).sort_index().fillna(0.0)

    if weight_smooth > 0:
        weights = weights.ewm(span=weight_smooth).mean()

    aligned_returns = returns.reindex(weights.index).fillna(0.0)
    gross_ret = (weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
    costs = apply_costs_from_weights(weights, slippage_bps=slippage_bps)
    net_ret = gross_ret - costs
    equity = (1.0 + net_ret).cumprod()
    equity.name = "equity"

    return equity, pair_positions, weights
