"""Applicability domain per entity (design §8).

Three criteria, each turned into a percentile among the entities of the same level:

1. support in the data: number of observed indicators;
2. latent support: mean distance of u_g to its ``k`` nearest well-observed entities at the same level (entity
   factors scaled to unit variance per dimension), as a percentile of the same quantity for the well-observed
   entities themselves;
3. fit: root mean squared standardised residual of the observed cells, as a percentile among entities.

An entity is ``outside`` if it has fewer than ``min_obs[0]`` observed indicators or a percentile above
``outside``, ``edge`` if it has fewer than ``min_obs[1]`` or a percentile above ``edge``, else ``in_domain``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regsim.models.base import ModelData
from regsim.uncertainty.conformal import model_sigma


def _percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    if ref.size == 0:
        return np.full(values.shape, np.nan)
    out = np.searchsorted(ref, values, side="right") / ref.size
    return np.where(np.isfinite(values), out, np.nan)


def applicability(
    model,
    data: ModelData,
    k: int = 5,
    min_obs: tuple[int, int] = (2, 5),
    edge: float = 0.95,
    outside: float = 0.99,
) -> pd.DataFrame:
    """Applicability-domain table (rows: entities) with the criteria, their percentiles and ``domain``."""
    level = data.levels()
    n_obs = data.mask.sum(axis=1)
    U = np.asarray(model.embed_entities(), dtype=float)
    U = (U - U.mean(axis=0)) / np.where(U.std(axis=0) > 0, U.std(axis=0), 1.0)
    P = model.predict()
    sd = model_sigma(model, P.shape)
    with np.errstate(all="ignore"):
        r = np.where(data.mask & np.isfinite(sd) & (sd > 0), ((data.Z - P) / sd) ** 2, np.nan)
        cnt = np.isfinite(r).sum(axis=1)
        resid = np.where(cnt > 0, np.sqrt(np.nansum(r, axis=1) / np.maximum(cnt, 1)), np.nan)
    knn = np.full(len(level), np.nan)
    knn_pct = np.full(len(level), np.nan)
    resid_pct = np.full(len(level), np.nan)
    for lv in np.unique(level):
        rows = np.flatnonzero(level == lv)
        support = rows[n_obs[rows] >= min_obs[1]]
        if support.size < 2:
            continue
        D = np.sqrt(((U[rows][:, None, :] - U[support][None, :, :]) ** 2).sum(axis=-1))
        D[rows[:, None] == support[None, :]] = np.inf
        kk = min(k, support.size - 1)
        knn[rows] = np.sort(D, axis=1)[:, :kk].mean(axis=1)
        knn_pct[rows] = _percentile(knn[rows], knn[support])
        resid_pct[rows] = _percentile(resid[rows], resid[support])
    worst = np.fmax(knn_pct, resid_pct)
    domain = np.where(
        (n_obs < min_obs[0]) | (worst > outside),
        "outside",
        np.where((n_obs < min_obs[1]) | (worst > edge), "edge", "in_domain"),
    )
    return pd.DataFrame(
        {
            "level": level,
            "n_observed": n_obs,
            "knn_distance": knn,
            "knn_percentile": knn_pct,
            "residual": resid,
            "residual_percentile": resid_pct,
            "domain": domain,
        },
        index=pd.Index(data.geo_ids, name="geo_id"),
    )
