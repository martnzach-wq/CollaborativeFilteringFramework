"""Intervention effects by counterfactual matrix completion (option C5).

An event table lists (geo_id, intervention, announcement_year, start_year, source). For
each treated entity, post-treatment cells of the outcome are masked and imputed by the
fitted regsim estimator (HierTypedMF or a baseline) using untreated entities from the
causal-lens analogue set; the effect is observed minus imputed, with intervals from
`regsim.uncertainty`. Placebo runs on untreated analogues give the reference
distribution.

The pre-treatment window ends at the announcement year, not the start year (M7 example:
Estonian motor vehicle tax in force from 1 January 2025; purchases were brought forward
into 2024).

Implementation: the default imputation is matrix completion with nuclear-norm penalty and
entity and year effects (fitted on the observed cells, then soft-impute of the residuals) (Athey, Bayati, Doudchenko, Imbens and Khosravi 2021,
doi:10.1080/01621459.2021.1891924) on the entity x year matrix of the outcome; any callable
``completed = model(Y)`` that fills the missing cells of a DataFrame can replace it. Intervals
and p-values come from in-space placebos: the same masking on untreated analogues, whose
observed-minus-imputed differences form the reference distribution (split-conformal in spirit,
as in `regsim.uncertainty.conformal`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator


class Event(BaseModel):
    """One intervention in one entity."""

    id: str
    geo_id: str
    intervention: str
    announcement_year: int
    start_year: int
    outcomes: list[str] = Field(default_factory=list)
    intensity: float | None = None
    source: str
    note: str | None = None

    @model_validator(mode="after")
    def _years(self) -> Event:
        if self.announcement_year > self.start_year:
            raise ValueError(f"{self.id}: announcement after start")
        return self


class EventTable(BaseModel):
    schema_version: str = "0.1"
    events: list[Event]


def load_events(path: str | Path) -> pd.DataFrame:
    """Event table from ``configs/events.yaml`` (validated); one row per event."""
    table = EventTable.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    return pd.DataFrame([e.model_dump() for e in table.events])


def _two_way(X: np.ndarray, obs: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Entity and year effects fitted on the observed cells of an unbalanced matrix (alternating means)."""
    mu = X[obs].mean()
    a = np.zeros(X.shape[0])
    b = np.zeros(X.shape[1])
    R = np.where(obs, X - mu, 0.0)
    n_row, n_col = obs.sum(axis=1).clip(min=1), obs.sum(axis=0).clip(min=1)
    for _ in range(n_iter):
        a = (np.where(obs, R - b[None, :], 0.0)).sum(axis=1) / n_row
        b = (np.where(obs, R - a[:, None], 0.0)).sum(axis=0) / n_col
    return mu + a[:, None] + b[None, :]


def mc_nnm(Y: pd.DataFrame, lam: float | None = None, max_iter: int = 500, tol: float = 1e-6) -> pd.DataFrame:
    """Matrix completion with nuclear-norm penalty and entity and year effects.

    ``Y`` is entity x year with NaN for the cells to impute. Entity and year effects are fitted on the
    observed cells; the residuals are completed by soft-impute (singular-value thresholding) with the
    penalty ``lam`` (default: a tenth of the largest singular value of the observed residuals).
    """
    X = Y.to_numpy(dtype=float)
    obs = np.isfinite(X)
    base = _two_way(X, obs)
    R = np.where(obs, X - base, 0.0)
    if lam is None:
        lam = 0.1 * np.linalg.svd(R, compute_uv=False)[0]
    L = np.zeros_like(R)
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(np.where(obs, R, L), full_matrices=False)
        L_new = (U * np.maximum(s - lam, 0.0)) @ Vt
        delta = np.linalg.norm(L_new - L) / max(np.linalg.norm(L), 1e-12)
        L = L_new
        if delta < tol:
            break
    return pd.DataFrame(np.where(obs, X, base + L), index=Y.index, columns=Y.columns)


def _two_sided(stat: float, null: pd.Series) -> float:
    """Two-sided placebo p-value of ``stat`` against the placebo averages ``null`` (same distribution as
    the interval, so that an interval excluding zero and p < alpha agree up to discreteness)."""
    null = null.dropna()
    if not len(null) or not np.isfinite(stat):
        return np.nan
    upper = (1 + (null - null.median() >= stat - null.median()).sum()) / (1 + len(null))
    lower = (1 + (null - null.median() <= stat - null.median()).sum()) / (1 + len(null))
    return float(min(1.0, 2 * min(upper, lower)))


def _wide(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return panel.pivot_table(index="geo_id", columns="year", values=outcome, aggfunc="mean")


def choose_analogues(
    wide: pd.DataFrame,
    treated: str,
    pre_years: list[int],
    k: int,
    exclude: set[str],
    lens: pd.DataFrame | None = None,
) -> list[str]:
    """``k`` untreated entities nearest to ``treated``: on the causal lens (``lens``: entity x indicator,
    e.g. region means of the outcome's causes) where given, else on the standardised pre-period path
    of the outcome. Entities need complete pre-period values."""
    pool = wide.loc[wide[pre_years].notna().all(axis=1)].index.difference(sorted(exclude | {treated}))
    if lens is not None:
        L = lens.reindex(pool.union([treated])).dropna()
        Z = (L - L.mean()) / L.std()
        d = ((Z.drop(index=treated, errors="ignore") - Z.loc[treated]) ** 2).mean(axis=1)
    else:
        P = wide.loc[pool.union([treated]), pre_years]
        P = P.sub(P.mean(axis=1), axis=0)  # level differences are absorbed by the entity effect
        d = ((P.drop(index=treated) - P.loc[treated]) ** 2).mean(axis=1)
    return list(d.sort_values().index[:k])


def _gaps(wide: pd.DataFrame, unit: str, donors: list[str], mask_from: int, model) -> pd.Series:
    Y = wide.loc[[unit, *donors]].copy()
    post = [c for c in Y.columns if c >= mask_from]
    truth = Y.loc[unit, post].copy()
    Y.loc[unit, post] = np.nan
    completed = model(Y)
    return truth - completed.loc[unit, post]


def effect(
    model,
    events: pd.DataFrame,
    outcome: str,
    analogues: int = 20,
    placebo: int = 200,
    seed: int = 0,
    panel: pd.DataFrame | None = None,
    alpha: float = 0.1,
    lens: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimated effect per treated entity and post year, with placebo p-values.

    Parameters
    ----------
    model : callable or None
        Imputation ``completed = model(Y)`` for an entity x year DataFrame with NaN cells; None uses
        `mc_nnm`.
    events : DataFrame
        From `load_events`: ``id``, ``geo_id``, ``announcement_year``, ``start_year``.
    outcome : str
        Indicator column of ``panel`` (transformed scale, as from `causal.data.annual_panel`).
    analogues : int
        Untreated donor entities per treated entity (`choose_analogues`).
    placebo : int
        Maximum number of placebo runs on untreated analogues (in-space placebos).
    panel : DataFrame
        Long panel with ``geo_id``, ``year`` and ``outcome``.
    lens : DataFrame, optional
        Causal-lens values (entity x indicator) for choosing analogues.

    Returns
    -------
    DataFrame
        One row per event and year from the announcement year: ``observed``, ``counterfactual``,
        ``effect`` (observed minus counterfactual), ``lower``/``upper`` (1 - ``alpha`` interval from the
        placebo gaps of that year), ``phase`` (anticipation before the start year, else treatment),
        ``p_value`` (two-sided, of the average effect over the treatment years against the placebo
        averages), ``pre_end`` (last pre-treatment year) and ``n_placebo``. ``attrs['summary']`` holds
        per event the average effect over the treatment years (``att``) with its interval from the
        placebo averages, separately for the anticipation years (announcement to start) and the treatment
        years.
    """
    if panel is None:
        raise ValueError("effect needs the long panel with the outcome")
    model = model or mc_nnm
    wide = _wide(panel, outcome)
    treated_ids = set(events.geo_id)
    rng = np.random.default_rng(seed)
    rows, summary = [], []
    for ev in events.itertuples():
        if ev.geo_id not in wide.index:
            continue
        mask_from = int(ev.announcement_year)  # the pre-period ends before the announcement year
        pre = [c for c in wide.columns if c < mask_from]
        post = [c for c in wide.columns if c >= mask_from]
        if len(pre) < 3 or not post or wide.loc[ev.geo_id, post].isna().all():
            continue
        donors = choose_analogues(wide, ev.geo_id, pre, analogues, treated_ids, lens)
        gap = _gaps(wide, ev.geo_id, donors, mask_from, model)
        pool = choose_analogues(wide, ev.geo_id, pre, max(placebo, analogues) + analogues, treated_ids, lens)
        pool = list(rng.permutation(pool)[:placebo])
        placebo_gaps = []
        for p in pool:  # each placebo unit is completed from the other analogues
            others = [d for d in pool if d != p][:analogues]
            placebo_gaps.append(_gaps(wide, p, others, mask_from, model))
        G = pd.DataFrame(placebo_gaps)
        treat_years = [y for y in post if y >= ev.start_year]
        p_value = (
            _two_sided(float(gap[treat_years].mean()), G[treat_years].mean(axis=1)) if treat_years else np.nan
        )
        for phase, years in (
            ("anticipation", [y for y in post if y < ev.start_year]),
            ("treatment", treat_years),
        ):  # average effect per phase, interval and p-value from the placebo averages
            years = [y for y in years if pd.notna(gap.get(y))]
            if not years:
                continue
            avg = float(gap[years].mean())
            mean_null = G[years].mean(axis=1)
            lo, hi = mean_null.quantile([alpha / 2, 1 - alpha / 2]) if len(G) else (np.nan, np.nan)
            p_phase = _two_sided(avg, mean_null)
            summary.append(
                {"event": ev.id, "geo_id": ev.geo_id, "phase": phase, "att": avg, "lower": avg - hi,
                 "upper": avg - lo, "p_value": p_phase, "years": f"{min(years)}-{max(years)}",
                 "pre_end": max(pre), "n_placebo": len(G)}
            )  # fmt: skip
        for y in post:
            observed = wide.loc[ev.geo_id, y]
            q_lo, q_hi = G[y].quantile([alpha / 2, 1 - alpha / 2]) if len(G) else (np.nan, np.nan)
            rows.append(
                {
                    "event": ev.id,
                    "geo_id": ev.geo_id,
                    "year": int(y),
                    "observed": observed,
                    "counterfactual": observed - gap[y],
                    "effect": gap[y],
                    "lower": gap[y] - q_hi,
                    "upper": gap[y] - q_lo,
                    "phase": "anticipation" if y < ev.start_year else "treatment",
                    "p_value": p_value,
                    "pre_end": max(pre),
                    "n_placebo": len(G),
                    "donors": ",".join(donors),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["summary"] = pd.DataFrame(summary)
    return out


ImputeFn = Callable[[pd.DataFrame], pd.DataFrame]
