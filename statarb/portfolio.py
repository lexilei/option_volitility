"""Portfolio aggregation and neutrality constraints."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .pairs import Pair


def aggregate_pair_weights(
    pairs: Iterable[Pair],
    pair_positions: Dict[str, float],
    betas_t: Dict[str, float],
    max_pair_w: float,
) -> pd.Series:
    weights: Dict[str, float] = {}
    for pair in pairs:
        key = f"{pair.y}-{pair.x}"
        signal = pair_positions.get(key, 0.0)
        # signal is ±1, scale by max_pair_w
        w_pair = float(np.clip(signal * max_pair_w, -max_pair_w, max_pair_w))
        beta_t = betas_t.get(key, pair.init_beta)
        weights[pair.y] = weights.get(pair.y, 0.0) + w_pair
        weights[pair.x] = weights.get(pair.x, 0.0) - w_pair * beta_t
    return pd.Series(weights, name="weight")


def neutralize_by_sector(weights: pd.Series, sector_map: Dict[str, str]) -> pd.Series:
    adj = weights.copy()
    sectors = {sector_map.get(sym, "UNKNOWN") for sym in weights.index}
    for sector in sectors:
        symbols = [s for s in weights.index if sector_map.get(s, "UNKNOWN") == sector]
        if not symbols:
            continue
        net = adj.loc[symbols].sum()
        adj.loc[symbols] -= net / max(1, len(symbols))
    return adj


def neutralize_beta(weights: pd.Series, betas: pd.Series, target_beta: float = 0.0) -> pd.Series:
    common = weights.index.intersection(betas.index)
    if common.empty:
        return weights
    port_beta = (weights.loc[common] * betas.loc[common]).sum()
    delta = port_beta - target_beta
    if abs(delta) < 1e-12:
        return weights
    # Simple rescale to remove beta; replace with constrained optimization later.
    adj = weights.copy()
    adj.loc[common] -= delta * betas.loc[common] / (betas.loc[common].pow(2).sum() + 1e-12)
    return adj


def apply_limits(weights: pd.Series, w_max: float = 0.02, gross_max: float = 1.0) -> pd.Series:
    clipped = weights.clip(lower=-w_max, upper=w_max)
    gross = clipped.abs().sum()
    if gross > gross_max and gross > 0:
        clipped = clipped * (gross_max / gross)
    return clipped
