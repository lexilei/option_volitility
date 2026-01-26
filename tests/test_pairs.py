import numpy as np
import pandas as pd

from statarb.pairs import select_pairs


def test_select_pairs_finds_cointegration():
    rng = np.random.default_rng(123)
    n = 300
    x = np.cumsum(rng.normal(0, 1, n)) + 100
    y = 2.0 * x + rng.normal(0, 0.5, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"AAA": y, "BBB": x}, index=dates)
    sector_map = {"AAA": "TECH", "BBB": "TECH"}

    pairs = select_pairs(
        prices=prices,
        sector_map=sector_map,
        train_window=252,
        corr_lookback=252,
        pval_thresh=0.1,
        max_pairs=5,
    )
    assert len(pairs) >= 1
