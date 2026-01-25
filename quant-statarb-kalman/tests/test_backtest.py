import numpy as np
import pandas as pd

from statarb.backtest import walk_forward_backtest


def test_walk_forward_backtest_runs():
    rng = np.random.default_rng(42)
    n = 120
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    x = base + rng.normal(0, 0.5, n)
    y = 1.2 * base + rng.normal(0, 0.5, n)
    spy = base + rng.normal(0, 0.2, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"AAA": y, "BBB": x, "SPY": spy}, index=idx)
    sector_map = {"AAA": "TECH", "BBB": "TECH", "SPY": "INDEX"}

    equity, pair_positions, weights = walk_forward_backtest(
        prices=prices,
        sector_map=sector_map,
        spy=prices["SPY"],
        train_window=60,
        test_step=10,
        max_pairs=5,
        pval_thresh=0.2,
        half_life_min=1,
        half_life_max=50,
        slippage_bps=0.0,
    )
    assert not equity.empty
    assert not weights.empty
    assert pair_positions.shape[0] == weights.shape[0]
