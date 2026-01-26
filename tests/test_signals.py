import pandas as pd

from statarb.signals import compute_zscore, generate_pair_positions


def test_generate_pair_positions_runs():
    s = pd.Series([1, 2, 3, 2, 1, 0, -1, -2, -3])
    z = compute_zscore(s, lookback=3).fillna(0.0)
    pos = generate_pair_positions(z)
    assert len(pos) == len(z)
