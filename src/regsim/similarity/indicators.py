"""Indicator similarity (design §5.2): I-model, I-observed, I-partial, I-predictive."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from regsim.models.base import ModelData


def i_model(model, data: ModelData, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Model-implied correlation of indicators across the entities ``rows`` (default: all).

    corr(k, l) = v_k^T S_U v_l / sqrt(v_k^T S_U v_k * v_l^T S_U v_l), with S_U the covariance of the entity
    factors of ``rows``; computing it for different levels or regions tests the stationarity of the relations.
    """
    rows = np.arange(model.U_.shape[0]) if rows is None else np.asarray(rows)
    S = np.cov(model.U_[rows], rowvar=False)
    V = model.embed_indicators()
    C = V @ S @ V.T
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return pd.DataFrame(C / np.outer(d, d), index=data.column_ids, columns=data.column_ids)


def i_observed(data: ModelData, rows: np.ndarray | None = None, min_pairs: int = 30) -> pd.DataFrame:
    """Pairwise-complete Spearman correlations with their number of entities and 95 % confidence intervals
    (Fisher z); pairs with fewer than ``min_pairs`` common entities are NaN. Long format."""
    rows = np.arange(data.Z.shape[0]) if rows is None else np.asarray(rows)
    X = np.where(data.mask, data.Z, np.nan)[rows]
    out = []
    p = X.shape[1]
    for a in range(p):
        for b in range(a + 1, p):
            ok = np.isfinite(X[:, a]) & np.isfinite(X[:, b])
            n = int(ok.sum())
            rho = lo = hi = np.nan
            if n >= min_pairs:
                rho = stats.spearmanr(X[ok, a], X[ok, b]).statistic
                z, se = np.arctanh(np.clip(rho, -0.999999, 0.999999)), 1.06 / np.sqrt(max(n - 3, 1))
                lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
            out.append((data.column_ids[a], data.column_ids[b], n, rho, lo, hi))
    return pd.DataFrame(out, columns=["indicator_a", "indicator_b", "n", "rho", "lo", "hi"])


def i_partial(model, data: ModelData, rows: np.ndarray | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Partial correlations from the graphical lasso on the model-implied covariance of the completed profiles.

    The model-implied covariance is the covariance of the completed profiles plus the residual variances;
    ``alpha`` is the graphical-lasso penalty (on the correlation scale).
    """
    from sklearn.covariance import graphical_lasso

    from regsim.similarity.entities import noise_variance

    rows = np.arange(data.Z.shape[0]) if rows is None else np.asarray(rows)
    P = model.predict()[rows]
    C = np.cov(P, rowvar=False) + np.diag(noise_variance(model, data))
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    _, prec = graphical_lasso(R, alpha=alpha, max_iter=200)
    dp = np.sqrt(np.diag(prec))
    pc = -prec / np.outer(dp, dp)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=data.column_ids, columns=data.column_ids)


def i_predictive(
    model_factory,
    data: ModelData,
    target: str,
    candidates: list[str] | None = None,
    frac: float = 0.3,
    seed: int = 0,
) -> pd.Series:
    """Reduction in held-out RMSE of ``target`` when one candidate indicator is revealed (design §5.2).

    A share ``frac`` of the entities that observe the target lose their target value and all candidates; each
    candidate is then revealed alone for these entities and the model refitted. Values are RMSE(without) -
    RMSE(with candidate): positive means the candidate carries information about the target.
    """
    rng = np.random.default_rng(seed)
    t = data.column_ids.index(target)
    candidates = candidates or [c for c in data.column_ids if c != target]
    cidx = [data.column_ids.index(c) for c in candidates]
    rows = np.flatnonzero(data.mask[:, t])
    test_rows = rng.choice(rows, max(1, int(frac * rows.size)), replace=False)
    base = data.mask.copy()
    base[np.ix_(test_rows, cidx + [t])] = False

    def err(mask):
        P = model_factory().fit(data.with_mask(mask)).predict()
        return float(np.sqrt(np.mean((P[test_rows, t] - data.Z[test_rows, t]) ** 2)))

    e0 = err(base)
    out = {}
    for c, j in zip(candidates, cidx, strict=True):
        m = base.copy()
        m[test_rows, j] = data.mask[test_rows, j]
        out[c] = e0 - err(m)
    return pd.Series(out).sort_values(ascending=False)
