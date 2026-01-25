"""Pair selection and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class Pair:
    y: str
    x: str
    sector: str
    init_beta: float
    half_life: int
    pvalue: float


def find_candidate_pairs(
    prices: pd.DataFrame,
    top_k_per_symbol: int = 5,
    corr_lookback: int = 252,
) -> List[Tuple[str, str]]:
    """Return candidate pairs by correlation prefilter."""
    if len(prices) < corr_lookback:
        raise ValueError("Not enough data for corr_lookback")
    corr = prices.tail(corr_lookback).pct_change().corr()
    pairs: List[Tuple[str, str]] = []
    for sym in corr.columns:
        top = corr[sym].drop(sym).nlargest(top_k_per_symbol).index
        for other in top:
            if (other, sym) not in pairs:
                pairs.append((sym, other))
    return pairs


def engle_granger_test(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """Return p-value and OLS hedge ratio using Engle-Granger test."""
    coint_t, pvalue, _ = coint(y, x)
    beta = np.polyfit(x.values, y.values, 1)[0]
    return float(pvalue), float(beta)


def estimate_half_life(spread: pd.Series) -> int:
    """Estimate half-life of mean reversion via AR(1)."""
    spread_lag = spread.shift(1).iloc[1:]
    spread_ret = spread.diff().iloc[1:]
    beta = np.polyfit(spread_lag.values, spread_ret.values, 1)[0]
    if beta >= 0:
        return 999
    halflife = -np.log(2) / beta
    return int(max(1, round(halflife)))


def select_pairs(
    prices: pd.DataFrame,
    sector_map: Dict[str, str],
    train_window: int = 252,
    corr_lookback: int = 252,
    min_half_life: int = 2,
    max_half_life: int = 20,
    pval_thresh: float = 0.05,
    max_pairs: int = 50,
) -> List[Pair]:
    """Select robust pairs within the same sector."""
    if len(prices) < train_window:
        raise ValueError("Not enough data for train_window")
    train = prices.tail(train_window)
    candidates = find_candidate_pairs(train, corr_lookback=corr_lookback)
    selected: List[Pair] = []
    for y, x in candidates:
        if sector_map.get(y) != sector_map.get(x):
            continue
        ys = train[y].dropna()
        xs = train[x].dropna()
        common = ys.index.intersection(xs.index)
        ys = ys.loc[common]
        xs = xs.loc[common]
        if len(common) < train_window * 0.9:
            continue
        pval, beta = engle_granger_test(ys, xs)
        if pval > pval_thresh:
            continue
        spread = ys - beta * xs
        half_life = estimate_half_life(spread)
        if not (min_half_life <= half_life <= max_half_life):
            continue
        selected.append(Pair(y=y, x=x, sector=sector_map[y], init_beta=beta, half_life=half_life, pvalue=pval))
        if len(selected) >= max_pairs:
            break
    return selected
