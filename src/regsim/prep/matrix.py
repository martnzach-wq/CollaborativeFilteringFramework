"""From the observation store to model-ready data (design §3)."""

from __future__ import annotations

import numpy as np

from regsim.geo.scheme import HierarchyGraph
from regsim.models.base import ModelData
from regsim.prep.transforms import forward, inverse, robust_scale
from regsim.registry import IndicatorRegistry
from regsim.schema import Lens
from regsim.store.store import ObservationStore

POPULATION_ID = "population"


def prepare(
    store: ObservationStore,
    registry: IndicatorRegistry,
    period: str,
    levels: list[int] | None = None,
    indicators: list[str] | None = None,
    require_population: bool = True,
    merge_copies: bool = True,
) -> ModelData:
    """Entity x indicator data for one period window, transformed and robust-scaled.

    Parameters
    ----------
    store, registry
        Observation store and indicator registry.
    period : str
        Period window, e.g. ``"2018/2022"``.
    levels : list of int, optional
        Levels of the rows; default all stored levels.
    indicators : list of str, optional
        Columns; default the registry's model indicators (one per ``same_as`` group).
    require_population : bool
        Keep only rows with a population value in the window (the prototype's unit set).
    merge_copies : bool
        Fill each column from its ``same_as`` copies by source precedence.

    Notes
    -----
    Each column is scaled by the median and IQR/1.349 of its transformed values at the indicator's first
    native level (all observed rows if none is observed there).
    """
    indicators = indicators or registry.model_indicators()
    canonical = registry.canonical() if merge_copies else None
    m = store.matrix(
        Lens(indicators=tuple(indicators)),
        index_levels=levels,
        period=period,
        canonical=canonical,
        precedence=registry.precedence,
        static=registry.static_indicators(),
    )
    keep = np.ones(len(m.geo_ids), dtype=bool)
    if require_population:
        pop = store.matrix(Lens(indicators=(POPULATION_ID,)), geo_ids=m.geo_ids, period=period)
        keep = pop.observed[:, 0]
    geo_ids = [g for g, k in zip(m.geo_ids, keep, strict=True) if k]
    X = m.X[keep]
    level = m.level[keep]
    graph = store.graph().subgraph(geo_ids)
    return from_matrix(X, geo_ids, level, indicators, registry, graph)


def from_matrix(
    X: np.ndarray,
    geo_ids: list[str],
    level: np.ndarray,
    indicators: list[str],
    registry: IndicatorRegistry,
    graph: HierarchyGraph,
) -> ModelData:
    """Transform and scale an original-scale matrix and attach the hierarchy."""
    Z = np.full(X.shape, np.nan)
    scalers = []
    meta = []
    for j, ind_id in enumerate(indicators):
        ind = registry.indicator(ind_id)
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
    W, _ = graph.aggregation_matrix(geo_ids, "pop")
    country = np.array([_top(graph, g) for g in geo_ids])
    return ModelData(
        Z=Z,
        mask=np.isfinite(Z),
        parent=graph.parent_index(geo_ids),
        agg=W,
        column_meta=meta,
        level=np.asarray(level, dtype=int),
        country=country,
        geo_ids=list(geo_ids),
        column_ids=list(indicators),
        scalers=scalers,
    )


def to_original(data: ModelData, Z: np.ndarray) -> np.ndarray:
    """Map model-scale values back to the original scale."""
    out = np.empty_like(Z, dtype=float)
    for j, (kind, loc, scale) in enumerate(data.scalers):
        out[:, j] = inverse(Z[:, j] * scale + loc, kind)
    return out


def _top(graph: HierarchyGraph, geo_id: str) -> str:
    anc = graph.ancestors(geo_id)
    return anc[-1] if anc else geo_id
