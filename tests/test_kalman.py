import numpy as np
import pandas as pd

from statarb.kalman import kalman_regression


def test_kalman_regression_tracks_beta():
    rng = np.random.default_rng(7)
    n = 200
    x = np.linspace(10, 20, n) + rng.normal(0, 0.2, n)
    y = 1.0 + 1.5 * x + rng.normal(0, 0.2, n)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    y = pd.Series(y, index=idx)
    x = pd.Series(x, index=idx)

    _, beta, spread = kalman_regression(y, x, R=1e-3, Q=1e-4)
    assert beta.iloc[-1] == beta.iloc[-1]  # not NaN
    assert abs(beta.iloc[-1] - 1.5) < 0.3
    assert spread.abs().mean() < 1.0
