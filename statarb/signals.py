"""Signal generation for spread trading."""
from __future__ import annotations

import pandas as pd


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std(ddof=0)
    z = (spread - mean) / std
    return z


def generate_pair_positions(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
    proportional: bool = False,
) -> pd.Series:
    pos = []
    current = 0.0
    for val in z.fillna(0.0):
        if current == 0:
            if val >= entry:
                current = -1.0
            elif val <= -entry:
                current = 1.0
        else:
            if abs(val) <= exit:
                current = 0.0
            elif abs(val) >= stop:
                current = 0.0
        if proportional and current != 0:
            # Scale position by z-score magnitude (capped at stop)
            scale = min(abs(val), stop) / entry
            pos.append(current / abs(current) * scale)
        else:
            pos.append(current)
    return pd.Series(pos, index=z.index, name="position")
