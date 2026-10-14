"""Recommender tasks P1-P7 (design §7) on a fitted ``HierTypedMF`` (or a wrapper such as ``Conformalised``).

Inputs and outputs are on the original scale of the indicators; intervals are central (1 - alpha) intervals,
mapped through the inverse transform. Every output marks which values are observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from regsim.models.base import ModelData
from regsim.prep.transforms import forward, inverse
from regsim.schema import Lens
from regsim.similarity.entities import s_profile


def _base(model):
    """The underlying fitted ``HierTypedMF`` of a wrapper."""
    while hasattr(model, "estimator_") or hasattr(model, "best_estimator_"):
        model = getattr(model, "estimator_", None) or model.best_estimator_
    return model


def _data(model) -> ModelData:
    return _base(model).data_


def _to_model(data: ModelData, j: int, x) -> np.ndarray:
    kind, loc, scale = data.scalers[j]
    return (forward(np.asarray(x, dtype=float), kind) - loc) / scale


def _to_original(data: ModelData, j: int, z) -> np.ndarray:
    kind, loc, scale = data.scalers[j]
    return inverse(np.asarray(z, dtype=float) * scale + loc, kind)


def _frame(data: ModelData, rows, cols, mu, sd, observed, alpha) -> pd.DataFrame:
    """Long table of values and intervals on the original scale (rows x cols of the model-scale arrays)."""
    z = norm.ppf(1 - alpha / 2)
    out = []
    for i, g in enumerate(rows):
        for j in cols:
            m, s = mu[i, j], sd[i, j]
            lo, hi = (m - z * s, m + z * s) if np.isfinite(s) else (np.nan, np.nan)
            out.append(
                {
                    "geo_id": g,
                    "indicator": data.column_ids[j],
                    "value": float(_to_original(data, j, m)),
                    "lower": float(_to_original(data, j, lo)) if np.isfinite(lo) else np.nan,
                    "upper": float(_to_original(data, j, hi)) if np.isfinite(hi) else np.nan,
                    "observed": bool(observed[i, j]),
                }
            )
    return pd.DataFrame(out)


def _sigma(model, shape) -> np.ndarray:
    if hasattr(model, "predict_std"):
        return np.asarray(model.predict_std(), dtype=float)
    return np.full(shape, np.nan)


# ---------------------------------------------------------------------------------------------------- P1
def fill_gaps(model, geo_id: str | None = None, indicator_id: str | None = None, alpha: float = 0.1):
    """P1: completed values with intervals for one entity, one indicator or everything (observed values are
    reported as observed, with zero-width intervals)."""
    data = _data(model)
    rows = [data.geo_ids.index(geo_id)] if geo_id else list(range(len(data.geo_ids)))
    cols = [data.column_ids.index(indicator_id)] if indicator_id else list(range(len(data.column_ids)))
    P, S = model.predict(), _sigma(model, data.Z.shape)
    obs = data.mask
    mu = np.where(obs, data.Z, P)
    sd = np.where(obs, 0.0, S)
    return _frame(data, [data.geo_ids[i] for i in rows], cols, mu[rows], sd[rows], obs[rows], alpha)


# ---------------------------------------------------------------------------------------------------- P2
def new_entity(
    model,
    values: dict[str, float],
    parent: str | None = None,
    side=None,
    name: str = "new",
    alpha: float = 0.1,
) -> pd.DataFrame:
    """P2: completed profile of a new entity from a few indicator values (original scale), an optional parent
    entity of the fitted data and side information (fold-in)."""
    base, data = _base(model), _data(model)
    p = len(data.column_ids)
    z = np.full((1, p), np.nan)
    mask = np.zeros((1, p), dtype=bool)
    for k, v in values.items():
        j = data.column_ids.index(k)
        z[0, j], mask[0, j] = _to_model(data, j, v), True
    parents = [[(data.geo_ids.index(parent), 1.0)]] if parent else None
    side_new = None if side is None else np.atleast_2d(side)
    P = base.fold_in(z, mask, parents, side_new)
    sd = _fold_in_std(base, base.fold_in_cov_)
    mu = np.where(mask, z, P)
    return _frame(data, [name], range(p), mu, np.where(mask, 0.0, sd), mask, alpha)


def _fold_in_std(base, covs: np.ndarray) -> np.ndarray:
    lay = base.layout_
    out = np.full((covs.shape[0], len(lay.kinds)), np.nan)
    for j in base.gauss_:
        phi = np.append(base.V_[lay.start[j]], 1.0)
        out[:, j] = np.sqrt(np.einsum("i,nij,j->n", phi, covs, phi) + base.sigma2_[j])
    return out


# ---------------------------------------------------------------------------------------------------- P3
def new_indicator(
    model, values: dict[str, float], transform: str = "identity", name: str = "new"
) -> pd.Series:
    """P3: a new continuous indicator observed on some entities, completed for all entities (indicator fold-in
    with the entity factors fixed). Values are robust-scaled internally after ``transform``."""
    from regsim.prep.transforms import robust_scale

    base, data = _base(model), _data(model)
    z = np.full(len(data.geo_ids), np.nan)
    for g, v in values.items():
        z[data.geo_ids.index(g)] = forward(np.asarray(v, dtype=float), transform)
    loc, scale = robust_scale(z)
    mask = np.isfinite(z)
    pred = base.fold_in_indicator(np.where(mask, (z - loc) / scale, 0.0), mask)
    return pd.Series(
        inverse(pred * scale + loc, transform), index=pd.Index(data.geo_ids, name="geo_id"), name=name
    )


# ---------------------------------------------------------------------------------------------------- P4
def vision_entity(model, base_geo: str, targets: dict[str, float], alpha: float = 0.1) -> pd.DataFrame:
    """P4: a fictive entity that starts from ``base_geo`` (its factors are the prior) and meets ``targets``
    (original scale); the other indicators move with the targets as the model's correlations imply.

    Returns, per indicator, the present value of the base entity, the vision value with interval and the change.
    """
    vision = new_entity(model, targets, parent=base_geo, name=f"vision:{base_geo}", alpha=alpha)
    present = fill_gaps(model, base_geo, alpha=alpha).set_index("indicator")
    out = vision.set_index("indicator")[["value", "lower", "upper", "observed"]].rename(
        columns={"value": "vision", "observed": "target"}
    )
    out.insert(0, "present", present.value)
    out.insert(1, "present_observed", present.observed)
    out["change"] = out.vision - out.present
    return out.reset_index()


# ---------------------------------------------------------------------------------------------------- P5
def analogues(
    model, profile, lens: Lens | None = None, k: int = 10, level: int | None = None
) -> pd.DataFrame:
    """P5: the ``k`` real entities nearest to ``profile`` under S-profile. ``profile`` is a geo id, a dict of
    original-scale values per indicator (e.g. the vision column of :func:`vision_entity`) or a model-scale
    vector of all columns."""
    data = _data(model)
    space = s_profile(model, data, lens)
    if isinstance(profile, str):
        q = model.predict()[data.geo_ids.index(profile)]
        level = int(data.levels()[data.geo_ids.index(profile)]) if level is None else level
    elif isinstance(profile, dict):
        q = np.array([_to_model(data, j, profile[c]) for j, c in enumerate(data.column_ids)], dtype=float)
    else:
        q = np.asarray(profile, dtype=float)
    y = (q[space.lens.cols] - space.mean) @ space.whitener
    rows = space.lens.rows
    if level is not None:
        rows = rows[data.levels()[rows] == level]
    if isinstance(profile, str):
        rows = rows[rows != data.geo_ids.index(profile)]
    d = np.sqrt(((space.Y[rows] - y) ** 2).sum(axis=1))
    order = np.argsort(d)[:k]
    return pd.DataFrame(
        {
            "geo_id": [data.geo_ids[i] for i in rows[order]],
            "level": data.levels()[rows[order]],
            "distance": d[order],
            "observed_share": data.mask[rows[order]][:, space.lens.cols].mean(axis=1),
        }
    )


# ---------------------------------------------------------------------------------------------------- P6
def next_indicator(
    model, geo_id: str, targets: list[str], candidates: list[str] | None = None
) -> pd.DataFrame:
    """P6: expected reduction of the posterior variance of ``targets`` (model scale, summed) if one candidate
    indicator were observed for ``geo_id`` (Laplace posterior, Sherman-Morrison update), ranked."""
    base, data = _base(model), _data(model)
    g = data.geo_ids.index(geo_id)
    lay = base.layout_
    C = base.cov_[g]
    phi = {j: np.append(base.V_[lay.start[j]], 1.0) for j in base.gauss_}
    tcols = [data.column_ids.index(t) for t in targets]
    if candidates is None:
        cand = [j for j in base.gauss_ if not data.mask[g, j] and j not in tcols]
    else:
        cand = [data.column_ids.index(c) for c in candidates]
    total = sum(float(phi[t] @ C @ phi[t]) for t in tcols if t in phi)
    rows = []
    for j in cand:
        if j not in phi:
            continue
        cj = C @ phi[j]
        denom = base.sigma2_[j] + float(phi[j] @ cj)
        red = sum(float(phi[t] @ cj) ** 2 / denom for t in tcols if t in phi)
        rows.append(
            {
                "indicator": data.column_ids[j],
                "variance_reduction": red,
                "share": red / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("variance_reduction", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------------------------------- P7
def level_transfer(model, indicator_id: str, to_level: int, alpha: float = 0.1) -> pd.DataFrame:
    """P7: estimates of ``indicator_id`` at ``to_level`` (constrained disaggregation: the model's aggregation
    constraint and reconciliation keep them consistent with observed coarse values), with the observed parent
    value and the aggregate of the estimated children for comparison."""
    data = _data(model)
    j = data.column_ids.index(indicator_id)
    rows = np.flatnonzero(data.levels() == to_level)
    out = fill_gaps(model, indicator_id=indicator_id, alpha=alpha)
    out = out[out.geo_id.isin([data.geo_ids[i] for i in rows])].reset_index(drop=True)
    parent = {data.geo_ids[i]: data.parent[i] for i in rows}
    out["parent"] = [", ".join(data.geo_ids[p] for p, _ in parent[g]) for g in out.geo_id]
    pv = {}
    for g, ps in parent.items():
        if len(ps) == 1 and data.mask[ps[0][0], j]:
            pv[g] = float(_to_original(data, j, data.Z[ps[0][0], j]))
    out["parent_observed"] = out.geo_id.map(pv)
    return out
