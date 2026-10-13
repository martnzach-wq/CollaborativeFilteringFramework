"""Geo-entity similarity (design §5.1): S-profile, S-latent, S-gower, percentile scores, contributions, intervals.

All measures are computed for a lens (indicator set and levels; decision D4: default all model indicators).

- S-profile: Mahalanobis-type distance of completed profiles, d(g, h) = ||W_K (zhat_g - zhat_h)|| with W_K the
  inverse square root of the covariance of the completed profiles on K at the lens levels (ridge-regularised).
  Every pair is compared on every lens indicator, and correlated indicators are not counted twice.
- S-latent: the same on the latent projection U V_K^T (no biases or residual terms).
- S-gower: Gower distance on observed values (min-max scaled, pairwise deletion) with the coverage ratio.

Scores are percentile ranks of a distance among all pairs at the same level (100 = most similar), not 1 - d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regsim.models.base import ModelData
from regsim.schema import Lens


@dataclass
class LensIndex:
    rows: np.ndarray  # row indices in the lens
    cols: np.ndarray  # column indices in the lens
    names: list[str]


def resolve_lens(data: ModelData, lens: Lens | None = None) -> LensIndex:
    """Rows and columns of ``data`` selected by ``lens``."""
    lens = lens or Lens()
    if lens.indicators is None:
        cols = np.arange(data.Z.shape[1])
    else:
        missing = [k for k in lens.indicators if k not in data.column_ids]
        if missing:
            raise KeyError(f"lens indicators not in the data: {missing}")
        cols = np.array([data.column_ids.index(k) for k in lens.indicators])
    level = data.levels()
    rows = np.flatnonzero(np.isin(level, lens.levels)) if lens.levels is not None else np.arange(len(level))
    return LensIndex(
        rows, cols, [data.column_ids[j] for j in cols] if data.column_ids else [str(j) for j in cols]
    )


class ProfileSpace:
    """Whitened completed profiles on a lens; base of S-profile and S-latent.

    Parameters
    ----------
    profiles : ndarray, shape (n, p)
        Completed profiles on the model scale (all entities, all columns).
    data : ModelData
        Data the profiles belong to (levels, labels, observed mask).
    lens : Lens, optional
        Lens; default all columns and levels.
    noise_var : ndarray, optional
        Residual variance per column (all columns), added to the covariance of the completed profiles to give
        the model-implied covariance.
    ridge : float
        Ridge added to the covariance, relative to its mean diagonal.
    """

    def __init__(
        self,
        profiles: np.ndarray,
        data: ModelData,
        lens: Lens | None = None,
        noise_var: np.ndarray | None = None,
        ridge: float = 1e-3,
    ):
        self.data = data
        self.lens = resolve_lens(data, lens)
        self.level = data.levels()
        X = profiles[:, self.lens.cols]
        R = X[self.lens.rows]
        self.mean = R.mean(axis=0)
        C = np.cov(R, rowvar=False) if R.shape[0] > 1 else np.eye(X.shape[1])
        C = np.atleast_2d(C)
        if noise_var is not None:
            C = C + np.diag(np.asarray(noise_var, dtype=float)[self.lens.cols])
        C = C + ridge * max(np.trace(C) / C.shape[0], 1e-12) * np.eye(C.shape[0])
        w, Q = np.linalg.eigh(C)
        self.whitener = Q @ np.diag(1.0 / np.sqrt(w)) @ Q.T
        self.precision = Q @ np.diag(1.0 / w) @ Q.T
        self.profiles = X
        self.Y = (X - self.mean) @ self.whitener
        self.explained = w[::-1] / w.sum()

    def distances(self, query: int | np.ndarray, targets: np.ndarray | None = None) -> np.ndarray:
        """Distances from row(s) ``query`` to ``targets`` (default: lens rows)."""
        targets = self.lens.rows if targets is None else np.asarray(targets)
        q = np.atleast_1d(query)
        diff = self.Y[q][:, None, :] - self.Y[targets][None, :, :]
        d = np.sqrt((diff**2).sum(axis=-1))
        return d[0] if np.ndim(query) == 0 else d

    def contributions(self, g: int, h: int) -> pd.Series:
        """Share of each lens indicator in the squared distance of (g, h): c_k = delta_k (C^-1 delta)_k / d^2.

        Shares sum to one; negative shares occur where correlated indicators differ in opposite directions.
        """
        delta = self.profiles[g] - self.profiles[h]
        c = delta * (self.precision @ delta)
        tot = c.sum()
        return pd.Series(c / tot if tot > 0 else np.zeros_like(c), index=self.lens.names)

    def pair_distribution(self, level: int, max_pairs: int = 200_000, seed: int = 0) -> np.ndarray:
        """Sorted distances of (sampled) pairs of lens rows at ``level``; reference for percentile scores."""
        rows = self.lens.rows[self.level[self.lens.rows] == level]
        rng = np.random.default_rng(seed)
        n = rows.size
        if n < 2:
            return np.array([])
        if n * (n - 1) // 2 <= max_pairs:
            i, j = np.triu_indices(n, k=1)
        else:
            i = rng.integers(0, n, max_pairs)
            j = rng.integers(0, n, max_pairs)
            keep = i != j
            i, j = i[keep], j[keep]
        d = np.sqrt(((self.Y[rows[i]] - self.Y[rows[j]]) ** 2).sum(axis=1))
        return np.sort(d)


def percentile_score(d: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Score 0-100: share of reference pairs that are farther apart than ``d`` (100 = closer than all)."""
    if reference.size == 0:
        return np.full(np.shape(d), np.nan)
    return 100.0 * (1.0 - np.searchsorted(reference, d, side="right") / reference.size)


def s_profile(model, data: ModelData, lens: Lens | None = None, ridge: float = 1e-3) -> ProfileSpace:
    """S-profile space from a fitted model's completed profiles and residual variances."""
    return ProfileSpace(model.predict(), data, lens, noise_variance(model, data), ridge)


def noise_variance(model, data: ModelData) -> np.ndarray:
    """Residual variance per column: ``sigma2_`` of HierTypedMF, ``scale_**2`` of baselines, else 1."""
    p = data.Z.shape[1]
    if hasattr(model, "sigma2_"):
        return np.array([model.sigma2_.get(j, 1.0) for j in range(p)])
    if getattr(model, "scale_", None) is not None:
        return np.asarray(model.scale_, dtype=float) ** 2
    return np.ones(p)


def s_latent(model, data: ModelData, lens: Lens | None = None, ridge: float = 1e-3) -> ProfileSpace:
    """S-latent space: latent projection U V_K^T of a fitted ``HierTypedMF``."""
    li = resolve_lens(data, lens)
    lay = model.layout_
    V = model.V_[[lay.start[j] for j in range(len(lay.kinds))]]
    proj = np.full(data.Z.shape, 0.0)
    proj[:, li.cols] = model.U_ @ V[li.cols].T
    return ProfileSpace(proj, data, lens, noise_variance(model, data), ridge)


def s_gower(
    data: ModelData,
    query: int,
    targets: np.ndarray | None = None,
    lens: Lens | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gower distance on observed values (min-max over observed cells, pairwise deletion).

    Returns distances and coverage ratios (share of lens indicators observed in both entities); distances are
    NaN where no lens indicator is shared.
    """
    li = resolve_lens(data, lens)
    targets = li.rows if targets is None else np.asarray(targets)
    mask = data.mask if mask is None else mask
    X = np.where(mask, data.Z, np.nan)[:, li.cols]
    with np.errstate(all="ignore"):
        lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    span = np.where(np.isfinite(hi - lo) & (hi > lo), hi - lo, 1.0)
    Xs = (X - lo) / span
    diff = np.abs(Xs[targets] - Xs[query])
    both = np.isfinite(diff)
    n = both.sum(axis=1)
    with np.errstate(all="ignore"):
        d = np.where(n > 0, np.nansum(diff, axis=1) / n, np.nan)
    return d, n / len(li.cols)


def nearest(
    space: ProfileSpace,
    geo_id: str,
    k: int = 10,
    samples: np.ndarray | None = None,
    same_level: bool = True,
    n_contrib: int = 3,
) -> pd.DataFrame:
    """The ``k`` most similar entities to ``geo_id`` with distance, percentile score, 90 % interval of the
    distance (from posterior ``samples`` of the profiles, shape (s, n, p)), observed share and top contributors."""
    data = space.data
    g = data.geo_ids.index(geo_id)
    rows = space.lens.rows
    if same_level:
        rows = rows[space.level[rows] == space.level[g]]
    rows = rows[rows != g]
    d = space.distances(g, rows)
    order = np.argsort(d)[:k]
    ref = space.pair_distribution(int(space.level[g]))
    out = pd.DataFrame(
        {
            "geo_id": [data.geo_ids[i] for i in rows[order]],
            "level": space.level[rows[order]],
            "distance": d[order],
            "score": percentile_score(d[order], ref),
            "observed_share": data.mask[rows[order]][:, space.lens.cols].mean(axis=1),
        }
    )
    if samples is not None:
        sd = []
        for s in samples:
            S = ProfileSpace.__new__(ProfileSpace)
            S.Y = (s[:, space.lens.cols] - space.mean) @ space.whitener
            sd.append(np.sqrt(((S.Y[rows[order]] - S.Y[g]) ** 2).sum(axis=1)))
        sd = np.array(sd)
        out["distance_lo"], out["distance_hi"] = np.percentile(sd, 5, axis=0), np.percentile(sd, 95, axis=0)
    out["top_contributors"] = [
        ", ".join(space.contributions(g, h).sort_values(ascending=False).index[:n_contrib])
        for h in rows[order]
    ]
    return out
