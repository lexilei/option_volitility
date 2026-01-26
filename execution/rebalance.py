"""Target weights to orders (stub)."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def targets_to_orders(
    target_weights: Dict[str, float],
    prices: pd.Series,
    current_positions: Dict[str, float],
    equity: float,
) -> Dict[str, int]:
    orders: Dict[str, int] = {}
    for sym, w in target_weights.items():
        target_value = w * equity
        price = float(prices.get(sym, 0.0))
        if price <= 0:
            continue
        target_shares = int(target_value / price)
        current = int(current_positions.get(sym, 0))
        delta = target_shares - current
        if delta != 0:
            orders[sym] = delta
    return orders
