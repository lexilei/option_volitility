"""Risk management checks (stub)."""
from __future__ import annotations

from typing import Dict


def check_risk(
    weights: Dict[str, float],
    gross_max: float,
    net_max: float,
    max_daily_loss_pct: float,
) -> bool:
    gross = sum(abs(w) for w in weights.values())
    net = abs(sum(weights.values()))
    if gross > gross_max or net > net_max:
        return False
    # Daily loss check requires live PnL; handled by caller when available.
    _ = max_daily_loss_pct
    return True
