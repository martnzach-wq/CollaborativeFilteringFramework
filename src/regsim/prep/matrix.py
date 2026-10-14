"""From the observation store to model-ready data (design §3)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from regsim.geo.nuts import LAU_LEVEL
from regsim.geo.scheme import HierarchyGraph
from regsim.models.base import ModelData
from regsim.prep.transforms import forward, inverse, robust_scale
from regsim.registry import IndicatorRegistry
from regsim.schema import DType, Lens
from regsim.store.store import ObservationStore

POPULATION_ID = "population"


def prepare(
    store: ObservationStore,
    registry: IndicatorRegistry,
    period: str | list[str],
    levels: list[int] | None = None,
    indicators: list[str] | None = None,
    require_population: bool = True,
    merge_copies: bool = True,
    side: list[str] | None = None,
    spatial: bool = False,
    gisco_dir: str = "data/raw/gisco",
) -> ModelData:
    """Entity x indicator data for one or more period windows, transformed and robust-scaled.

    Parameters
    ----------
    store, registry
        Observation store and indicator registry.
    period : str or list of str
        Period window(s), e.g. ``"2018/2022"``. Several windows give geo-time entities: one row per entity and
        window (``geo_id@window``), parents within the same window and ``time_pairs`` linking consecutive
        windows (design §4.3).
    levels : list of int, optional
        Levels of the rows; default all stored levels.
    indicators : list of str, optional
        Columns; default the registry's model indicators (one per ``same_as`` group).
    require_population : bool
        Keep only rows with a population value in the window (the prototype's unit set).
    merge_copies : bool
        Fill each column from its ``same_as`` copies by source precedence.
    side : list of str, optional
        Side information per row from the geo registry: ``log_area``, ``log_population``, ``log_density``
        (standardised).
    spatial : bool
        Attach the adjacency of rows at the same level from the GISCO boundaries (NUTS rows only).

    Notes
    -----
    Continuous columns are scaled by the median and IQR/1.349 of their transformed values at the indicator's
    first native level (all observed rows if none is observed there). Ordinal and categorical columns keep
    their integer codes.
    """
    periods = [period] if isinstance(period, str) else list(period)
    indicators = indicators or registry.model_indicators()
    canonical = registry.canonical() if merge_copies else None
    blocks = []
    for w in periods:
        m = store.matrix(
            Lens(indicators=tuple(indicators)),
            index_levels=levels,
            period=w,
            canonical=canonical,
            precedence=registry.precedence,
            static=registry.static_indicators(),
        )
        keep = np.ones(len(m.geo_ids), dtype=bool)
        if require_population:
            pop = store.matrix(Lens(indicators=(POPULATION_ID,)), geo_ids=m.geo_ids, period=w)
            units = store.units().set_index("geo_id")
            # LAUs have no Eurostat population series; their population comes from the LAU list
            listed = units.population.reindex(m.geo_ids).notna().to_numpy() & (m.level >= LAU_LEVEL)
            keep = pop.observed[:, 0] | listed
        blocks.append((w, [g for g, k in zip(m.geo_ids, keep, strict=True) if k], m.X[keep], m.level[keep]))
    base_ids = [g for _, ids, _, _ in blocks for g in ids]
    graph = store.graph().subgraph(set(base_ids))
    multi = len(periods) > 1
    data = from_matrix(
        np.vstack([b[2] for b in blocks]),
        base_ids,
        np.concatenate([b[3] for b in blocks]),
        indicators,
        registry,
        graph,
        period=np.concatenate([[w] * len(ids) for w, ids, _, _ in blocks]) if multi else None,
    )
    if side:
        data.side, data.side_names = _side(store, base_ids, side), list(side)
    if spatial:
        data.adjacency = _adjacency(base_ids, data.period, gisco_dir)
    return data


def from_matrix(
    X: np.ndarray,
    geo_ids: list[str],
    level: np.ndarray,
    indicators: list[str],
    registry: IndicatorRegistry,
    graph: HierarchyGraph,
    period: np.ndarray | None = None,
) -> ModelData:
    """Transform and scale an original-scale matrix and attach the hierarchy.

    With ``period`` given, rows are geo-time entities: row labels become ``geo_id@period``, hierarchy and
    aggregation link rows of the same period, and ``time_pairs`` link the same entity in consecutive periods.
    """
    Z = np.full(X.shape, np.nan)
    scalers = []
    meta = []
    for j, ind_id in enumerate(indicators):
        ind = registry.indicator(ind_id)
        if ind.dtype in (DType.ORDINAL, DType.CATEGORICAL):
            Z[:, j] = X[:, j]
            scalers.append(("identity", 0.0, 1.0))
            meta.append(ind)
            continue
        kind = ind.transform.value
        t = forward(X[:, j], kind)
        native = [int(x) for x in ind.native_levels]
        ref = t[np.isin(level, native) & np.isfinite(t)] if native else t[np.isfinite(t)]
        if ref.size < 3:
            ref = t[np.isfinite(t)]
        loc, scale = robust_scale(ref) if ref.size else (0.0, 1.0)
        Z[:, j] = (t - loc) / scale
        scalers.append((kind, loc, scale))
        meta.append(ind)
    rows = _rows(geo_ids, period)
    n = len(geo_ids)
    parent = [[] for _ in range(n)]
    mats = {k: sparse.lil_matrix((n, n)) for k in ("pop", "area", "membership")}
    cover = np.zeros(n)
    for idx in rows.values():
        ids = [geo_ids[i] for i in idx]
        sub = graph.parent_index(ids)
        for local, i in enumerate(idx):
            parent[i] = [(idx[p], w) for p, w in sub[local]]
        for kind, M in mats.items():
            W, _ = graph.aggregation_matrix(ids, kind)
            W = W.tocoo()
            M[np.asarray(idx)[W.row], np.asarray(idx)[W.col]] = W.data
        cover[idx] = _coverage(graph, ids)
    country = np.array([_top(graph, g) for g in geo_ids])
    labels = list(geo_ids) if period is None else [f"{g}@{t}" for g, t in zip(geo_ids, period, strict=True)]
    return ModelData(
        Z=Z,
        mask=np.isfinite(Z),
        parent=parent,
        agg=mats["pop"].tocsr(),
        column_meta=meta,
        level=np.asarray(level, dtype=int),
        country=country,
        geo_ids=labels,
        column_ids=list(indicators),
        scalers=scalers,
        agg_area=mats["area"].tocsr(),
        agg_sum=mats["membership"].tocsr(),
        agg_cover=cover,
        period=None if period is None else np.asarray(period),
        time_pairs=None if period is None else _time_pairs(geo_ids, period),
    )


def _rows(geo_ids: list[str], period: np.ndarray | None) -> dict:
    if period is None:
        return {None: list(range(len(geo_ids)))}
    out: dict = {}
    for i, t in enumerate(period):
        out.setdefault(t, []).append(i)
    return out


def _time_pairs(geo_ids: list[str], period: np.ndarray) -> np.ndarray:
    order = sorted(set(period))
    pos = {(g, t): i for i, (g, t) in enumerate(zip(geo_ids, period, strict=True))}
    pairs = [
        (pos[(g, a)], pos[(g, b)])
        for a, b in zip(order, order[1:], strict=False)
        for g in set(geo_ids)
        if (g, a) in pos and (g, b) in pos
    ]
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def _coverage(graph: HierarchyGraph, ids: list[str]) -> np.ndarray:
    """Share of each unit's population covered by its children among ``ids`` (0 for units without children)."""
    present = set(ids)
    out = np.zeros(len(ids))
    for i, g in enumerate(ids):
        pop = graph.units[g].population
        kids = [(c, graph.membership(c, g)) for c, _ in graph.children(g) if c in present]
        if not kids or not pop:
            continue
        out[i] = sum(m * (graph.units[c].population or 0.0) for c, m in kids) / pop
    return out


def _side(store: ObservationStore, geo_ids: list[str], names: list[str]) -> np.ndarray:
    units = store.units().set_index("geo_id").reindex(geo_ids)
    area = units.area_km2.astype(float).to_numpy()
    pop = units.population.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            "log_area": np.log(area),
            "log_population": np.log(pop),
            "log_density": np.log(pop / area),
        }
    X = np.column_stack([feats[n] for n in names])
    X[~np.isfinite(X)] = np.nan
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)
    return np.where(np.isfinite(X), X, 0.0)


def _adjacency(geo_ids: list[str], period: np.ndarray | None, gisco_dir: str):
    from regsim.geo.nuts import adjacency_pairs
    from regsim.geo.scheme import split_geo_id

    schemes = {split_geo_id(g)[0] for g in geo_ids}
    pairs = set()
    for scheme in schemes:
        if scheme.startswith("NUTS"):
            pairs |= adjacency_pairs(scheme.removeprefix("NUTS"), gisco_dir)
    n = len(geo_ids)
    rows = _rows(geo_ids, period)
    A = sparse.lil_matrix((n, n))
    for idx in rows.values():
        pos = {split_geo_id(geo_ids[i])[1]: i for i in idx}
        for a, b in pairs:
            if a in pos and b in pos:
                A[pos[a], pos[b]] = A[pos[b], pos[a]] = 1.0
    return A.tocsr()


def to_original(data: ModelData, Z: np.ndarray) -> np.ndarray:
    """Map model-scale values back to the original scale."""
    out = np.empty_like(Z, dtype=float)
    for j, (kind, loc, scale) in enumerate(data.scalers):
        out[:, j] = inverse(Z[:, j] * scale + loc, kind)
    return out


def _top(graph: HierarchyGraph, geo_id: str) -> str:
    anc = graph.ancestors(geo_id)
    return anc[-1] if anc else geo_id
