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


# ---- compositions (shares summing to one) of entities without observations -------------------------------


def _clr(x: np.ndarray) -> np.ndarray:
    lx = np.log(np.clip(x, 1e-6, None))
    return lx - lx.mean(axis=-1, keepdims=True)


def _closure(c: np.ndarray) -> np.ndarray:
    e = np.exp(c - c.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class CompositionEstimate:
    estimate: pd.DataFrame  # target x part, shares
    lower: pd.DataFrame  # weighted 10 % / 90 % range of the donors, closed
    upper: pd.DataFrame
    weights: pd.DataFrame  # target x donor kernel weights
    bandwidth: float
    loo_error: float  # mean Aitchison distance of the leave-one-out prediction of the donors
    loo_error_mean: float  # the same for the unweighted mean of the other donors


def estimate_composition(
    shares: pd.DataFrame,
    features: pd.DataFrame,
    targets: list[str],
    bandwidths: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0, np.inf),
) -> CompositionEstimate:
    """Composition of ``targets`` as the kernel-weighted mean of the donors' ``shares`` in centred log-ratio
    space; weights exp(-(d/h)^2) on standardised ``features`` (donors and targets, same columns). The
    bandwidth h minimises the leave-one-out Aitchison error of the donors; h = inf is the unweighted mean."""
    donors = list(shares.index)
    shares = shares.div(shares.sum(axis=1), axis=0)
    features = features[~features.index.duplicated()]
    f = features.loc[donors + [t for t in targets if t not in donors]]
    f = f.fillna(f.loc[donors].mean())  # a missing feature of a target takes the donors' mean
    z = (f - f.mean()) / f.std(ddof=0).replace(0, 1)
    zd, clr = z.loc[donors].to_numpy(), _clr(shares.to_numpy())

    def kernel(a: np.ndarray, b: np.ndarray, h: float) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.ones_like(d2) if np.isinf(h) else np.exp(-d2 / h**2)

    def loo(h: float) -> float:
        w = kernel(zd, zd, h)
        np.fill_diagonal(w, 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        return float(np.sqrt(((w @ clr - clr) ** 2).sum(axis=1)).mean())

    errors = {h: loo(h) for h in bandwidths}
    h = min(errors, key=errors.get)
    w = kernel(z.loc[targets].to_numpy(), zd, h)
    w = w / w.sum(axis=1, keepdims=True)
    est = _closure(w @ clr)
    lo, hi = np.empty_like(est), np.empty_like(est)
    for i in range(len(targets)):
        order = np.argsort(clr, axis=0)
        for j in range(clr.shape[1]):
            c = np.cumsum(w[i, order[:, j]])
            lo[i, j] = clr[order[np.searchsorted(c, 0.1), j], j]
            hi[i, j] = clr[order[min(np.searchsorted(c, 0.9), len(c) - 1), j], j]
    frame = lambda a: pd.DataFrame(a, index=targets, columns=shares.columns)  # noqa: E731
    return CompositionEstimate(
        estimate=frame(est),
        lower=frame(np.minimum(_closure(lo), est)),
        upper=frame(np.maximum(_closure(hi), est)),
        weights=pd.DataFrame(w, index=targets, columns=donors),
        bandwidth=float(h),
        loo_error=errors[h],
        loo_error_mean=errors[np.inf] if np.inf in errors else float("nan"),
    )
