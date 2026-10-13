"""Transfer-specific similarity (design §5.3): learned non-negative indicator weights for a target quantity.

For a target quantity y known on some entities, weights w >= 0 over the lens indicators define the distance
d_w(g, h)^2 = sum_k w_k (z_gk - z_hk)^2 on completed, robust-scaled profiles. The weights minimise the
leave-one-out error of the kernel-weighted neighbour prediction
    yhat_g = sum_{h != g} exp(-d_w(g, h)^2) y_h / sum_{h != g} exp(-d_w(g, h)^2)
plus lam * ||w - w0||^2, which pulls the weights towards the default lens (equal weights w0). With few labelled
entities the result is a diagnostic: bootstrap intervals show which weights are determined by the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class TransferWeights:
    weights: pd.Series
    lower: pd.Series
    upper: pd.Series
    loo_rmse: float
    loo_rmse_default: float

    def distance(self, profiles: np.ndarray, g: int, h: np.ndarray) -> np.ndarray:
        w = self.weights.to_numpy()
        return np.sqrt(((profiles[h] - profiles[g]) ** 2 * w).sum(axis=1))


def _loo(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    diff2 = (X[:, None, :] - X[None, :, :]) ** 2
    D = diff2 @ w
    K = np.exp(-(D - D.min(axis=1, keepdims=True)))
    np.fill_diagonal(K, 0.0)
    s = K.sum(axis=1)
    pred = (K @ y) / np.where(s > 0, s, 1.0)
    return float(np.mean((pred - y) ** 2)), pred


def learn_transfer_weights(
    profiles: np.ndarray,
    labelled: np.ndarray,
    target: np.ndarray,
    names: list[str],
    lam: float = 0.1,
    n_boot: int = 200,
    seed: int = 0,
) -> TransferWeights:
    """Learn transfer weights.

    Parameters
    ----------
    profiles : ndarray, shape (n, p)
        Completed profiles on the lens (model scale).
    labelled : ndarray of int
        Rows with a known target value.
    target : ndarray
        Target values of ``labelled`` (standardised internally).
    names : list of str
        Lens indicator names.
    lam : float
        Pull towards equal weights.
    n_boot : int
        Bootstrap resamples of the labelled entities for the intervals.
    """
    X = np.asarray(profiles, dtype=float)[labelled]
    y = np.asarray(target, dtype=float)
    y = (y - y.mean()) / (y.std() or 1.0)
    p = X.shape[1]
    w0 = np.full(p, 1.0 / p)

    def fit(Xb, yb):
        def obj(theta):
            w = np.exp(theta)
            return _loo(w, Xb, yb)[0] + lam * np.sum((w - w0) ** 2) * p

        res = minimize(obj, np.log(w0), method="L-BFGS-B")
        return np.exp(res.x)

    w = fit(X, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot if len(y) > 3 else 0):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(idx)) < 3:
            continue
        boots.append(fit(X[idx], y[idx]))
    boots = np.array(boots) if boots else np.array([w])
    return TransferWeights(
        weights=pd.Series(w, index=names),
        lower=pd.Series(np.percentile(boots, 5, axis=0), index=names),
        upper=pd.Series(np.percentile(boots, 95, axis=0), index=names),
        loo_rmse=float(np.sqrt(_loo(w, X, y)[0])),
        loo_rmse_default=float(np.sqrt(_loo(w0, X, y)[0])),
    )
