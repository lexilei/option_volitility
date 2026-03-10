"""Kalman filter regression for dynamic hedge ratio."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def kalman_regression(
    y: pd.Series,
    x: pd.Series,
    R: float,
    Q: float,
    state_init: np.ndarray | None = None,
    P_init: np.ndarray | None = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return alpha_t, beta_t, spread, and normalized innovation z-score.

    Optionally accepts initial state/covariance to warm-start across windows.
    """
    if R <= 0 or Q <= 0:
        raise ValueError("R and Q must be positive")
    idx = y.index.intersection(x.index)
    yv = y.loc[idx]
    xv = x.loc[idx]
    mask = ~(yv.isna() | xv.isna())
    yv = yv.loc[mask].values
    xv = xv.loc[mask].values
    idx = idx[mask]

    n = len(idx)
    if n < 5:
        raise ValueError("Not enough overlapping data points")
    alpha = np.zeros(n)
    beta = np.zeros(n)
    innovation_z = np.zeros(n)

    # State: [alpha, beta]
    if state_init is not None:
        state = state_init.copy()
    else:
        state = np.array([0.0, np.polyfit(xv, yv, 1)[0]])
    if P_init is not None:
        P = P_init.copy()
    else:
        P = np.eye(2)
    Qm = np.eye(2) * Q
    Rm = R

    for i in range(n):
        # Predict
        P = P + Qm
        yhat = state[0] + state[1] * xv[i]
        err = yv[i] - yhat
        H = np.array([1.0, xv[i]])
        S = H @ P @ H.T + Rm
        K = P @ H.T / (S + 1e-12)
        state = state + K * err
        P = P - np.outer(K, H) @ P
        alpha[i] = state[0]
        beta[i] = state[1]
        innovation_z[i] = err / np.sqrt(S + 1e-12)

    alpha_s = pd.Series(alpha, index=idx, name="alpha")
    beta_s = pd.Series(beta, index=idx, name="beta")
    spread = y.loc[idx] - (alpha_s + beta_s * x.loc[idx])
    spread.name = "spread"
    innov_z = pd.Series(innovation_z, index=idx, name="innovation_z")
    return alpha_s, beta_s, spread, innov_z
