"""Trading cost model."""
from __future__ import annotations

import pandas as pd


def apply_costs(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    slippage_bps: float,
    commission_per_share: float,
) -> pd.Series:
    """Return daily cost series based on turnover (placeholder)."""
    deltas = positions.diff().abs().fillna(0.0)
    notional = deltas * prices.reindex_like(deltas)
    slippage = notional.sum(axis=1) * (slippage_bps / 1e4)
    commission = deltas.sum(axis=1) * commission_per_share
    return slippage + commission


def apply_costs_from_weights(weights: pd.DataFrame, slippage_bps: float) -> pd.Series:
    """Return daily cost series using weight turnover (fraction of equity)."""
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return turnover * (slippage_bps / 1e4)
