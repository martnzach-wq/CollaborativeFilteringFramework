"""Annual panels from the observation store for the causal layer (M7).

Values are read on the original scale and transformed at model time with the transform of the
indicator registry. The parent group of a region (its country) comes from the hierarchy graph of
the `GeoScheme`, never from code prefixes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from regsim.prep.transforms import forward


def annual_panel(
    store,
    registry=None,
    indicators: Iterable[str] | None = None,
    level: int = 2,
    since: int | None = None,
    until: int | None = None,
    transform: bool = True,
    scenario: str = "observed",
    transforms: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Long panel with one row per entity-year at ``level``.

    Parameters
    ----------
    store : ObservationStore
        Store with annual observations (periods ``"YYYY"``).
    registry : IndicatorRegistry, optional
        Source of the transforms and of the ``same_as`` copies, which fill the gaps of the indicator
        that enters the model; without it values stay on the original scale.
    indicators : iterable of str, optional
        Indicator ids (default: all in the store).
    level : int
        Level of the entities (0 = countries, 2 = NUTS-2).
    since, until : int, optional
        First and last year.
    transform : bool
        Apply the registry transform (log, log1p, logit) to each indicator.
    transforms : dict, optional
        Indicator id -> transform replacing the registry transform (``causal.transforms`` of the
        config; e.g. log for cars per 1,000, so that log-log coefficients are elasticities).

    Returns
    -------
    DataFrame
        Columns ``geo_id``, ``year``, ``parent_id`` (level-0 ancestor, the entity itself at level 0)
        and one column per indicator.
    """
    units = store.units()
    ids = units.loc[units.level == level, "geo_id"].tolist()
    obs = store.query(
        "SELECT geo_id, indicator_id, period, value FROM observations WHERE scenario = ?", [scenario]
    )
    obs = obs[obs.geo_id.isin(set(ids)) & obs.period.astype(str).str.fullmatch(r"\d{4}")]
    if indicators is not None:
        obs = obs[obs.indicator_id.isin(set(indicators))]
    obs = obs.assign(year=obs.period.astype(int))
    if since is not None:
        obs = obs[obs.year >= since]
    if until is not None:
        obs = obs[obs.year <= until]
    wide = obs.pivot_table(index=["geo_id", "year"], columns="indicator_id", values="value", aggfunc="mean")
    wide.columns.name = None
    wide = wide.reset_index()
    if registry is not None:  # copies (same_as) fill the gaps of the indicator that enters the model
        for copy, canon in registry.canonical().items():
            if copy != canon and copy in wide.columns:
                wide[canon] = wide[canon].fillna(wide[copy]) if canon in wide.columns else wide[copy]
                wide = wide.drop(columns=copy)
    if transform:
        for c in wide.columns[2:]:
            kind = (transforms or {}).get(c) or (
                str(registry[c].transform) if registry is not None and c in registry.entries else None
            )
            if kind:
                wide[c] = forward(wide[c].to_numpy(), kind)
    graph = store.graph()
    country = {}
    for g in wide.geo_id.unique():
        top = [a for a in graph.ancestors(g) if graph.level(a) == 0]
        country[g] = g if level == 0 else (top[0] if top else None)
    wide.insert(2, "parent_id", wide.geo_id.map(country))
    return wide.sort_values(["geo_id", "year"]).reset_index(drop=True)


def region_means(panel: pd.DataFrame, indicators: Iterable[str], min_years: int = 5) -> pd.DataFrame:
    """Mean of each indicator over the years of each entity; entities with fewer than ``min_years``
    observed years of an indicator get NaN for it."""
    cols = [c for c in indicators if c in panel.columns]
    g = panel.groupby("geo_id")
    means = g[cols].mean()
    counts = g[cols].count()
    means = means.where(counts >= min_years, np.nan)
    means.insert(0, "parent_id", g["parent_id"].first())
    return means
