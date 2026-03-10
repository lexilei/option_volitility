"""Performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, freq: int = 252) -> float:
    vol = returns.std(ddof=0)
    if vol == 0:
        return 0.0
    return float(np.sqrt(freq) * returns.mean() / vol)


def sortino(returns: pd.Series, freq: int = 252) -> float:
    downside = returns[returns < 0].std(ddof=0)
    if downside == 0:
        return 0.0
    return float(np.sqrt(freq) * returns.mean() / downside)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())
