"""Within-region panel estimation of causal edges (option C2).

Reference implementation from the M7 analysis (causal_test_notes.md). The model for an
edge X -> Y on a long panel of entity-years is

    dY_t = b0 dX_t + b1 dX_{t-1} + rho dY_{t-1} + a_g + d_{c,t} + e_gt

with entity effects a_g and parent-by-year effects d_{c,t} (country x year for NUTS-2;
year effects for national panels). The reported effect is the cumulative short-run
effect b0 + b1, standardised by the demeaned standard deviations of dX and dY, with
cluster-robust standard errors by entity. The long-run effect is (b0 + b1) / (1 - rho).

Parent-by-year groups come from the hierarchy graph of the `GeoScheme`, never from code
prefixes (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EdgeEstimate:
    """Result of `fit_edge`."""

    cause: str
    effect: str
    beta_std: float
    t: float
    ci95_std: tuple[float, float]
    beta_raw: float
    se_raw: float
    long_run_raw: float
    se_long_run_raw: float
    rho: float
    n_obs: int
    n_entities: int


def demean(df: pd.DataFrame, keys: list[str], cols: list[str], tol: float = 1e-10, max_iter: int = 200) -> pd.DataFrame:
    """Remove several sets of fixed effects by alternating projections."""
    X = df[cols].astype(float).copy()
    for _ in range(max_iter):
        X_old = X.copy()
        for k in keys:
            X = X - X.groupby(df[k]).transform("mean")
        if float(np.nanmax(np.abs((X - X_old).to_numpy()))) < tol:
            break
    return X


def _prepare(panel: pd.DataFrame, cause: str, effect: str, entity: str, time: str, group: str | None) -> pd.DataFrame:
    d = panel[[entity, time, cause, effect] + ([group] if group else [])].sort_values([entity, time]).copy()
    g = d.groupby(entity)
    d["dy"] = g[effect].diff()
    d["dx0"] = g[cause].diff()
    d["dx1"] = d.groupby(entity)["dx0"].shift(1)
    d["dy_l"] = d.groupby(entity)["dy"].shift(1)
    d = d.dropna(subset=["dy", "dx0", "dx1", "dy_l"]).copy()
    d["fe_time"] = d[group].astype(str) + "_" + d[time].astype(str) if group else d[time].astype(str)
    return d


def fit_edge(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
    min_obs: int = 60,
    min_entities: int = 12,
) -> EdgeEstimate | None:
    """Estimate the cumulative within-entity effect of `cause` on `effect`.

    Parameters
    ----------
    panel
        Long table with one row per entity-year; values already transformed according to
        the indicator registry (log, log1p, identity).
    group
        Column with the parent entity used for parent-by-year effects; `None` gives plain
        year effects.

    Returns
    -------
    EdgeEstimate or None if the panel is too small.
    """
    d = _prepare(panel, cause, effect, entity, time, group)
    if len(d) < min_obs or d[entity].nunique() < min_entities:
        return None
    cols = ["dy", "dx0", "dx1", "dy_l"]
    Xd = demean(d, [entity, "fe_time"], cols)
    X = Xd[["dx0", "dx1", "dy_l"]].to_numpy()
    y = Xd["dy"].to_numpy()
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    res = y - X @ b
    meat = np.zeros((3, 3))
    for idx in d.groupby(entity).indices.values():
        s = X[idx].T @ res[idx]
        meat += np.outer(s, s)
    G = d[entity].nunique()
    V = XtX @ meat @ XtX * G / (G - 1)
    c = np.array([1.0, 1.0, 0.0])
    cum, se = float(c @ b), float(np.sqrt(c @ V @ c))
    rho = float(b[2])
    grad = np.array([1 / (1 - rho), 1 / (1 - rho), cum / (1 - rho) ** 2])
    sx, sy = float(Xd["dx0"].std()), float(Xd["dy"].std())
    return EdgeEstimate(
        cause=cause,
        effect=effect,
        beta_std=cum * sx / sy,
        t=cum / se,
        ci95_std=((cum - 1.96 * se) * sx / sy, (cum + 1.96 * se) * sx / sy),
        beta_raw=cum,
        se_raw=se,
        long_run_raw=cum / (1 - rho),
        se_long_run_raw=float(np.sqrt(grad @ V @ grad)),
        rho=rho,
        n_obs=len(d),
        n_entities=G,
    )


def screen(panel: pd.DataFrame, indicators: list[str], alpha: float = 0.05, **kwargs) -> pd.DataFrame:
    """Fit all ordered pairs and flag those above a Bonferroni threshold.

    Pairs flagged here are candidates only; `graph.classify_mechanism` removes accounting
    identities, shared denominators and shared survey bases before they go to the
    literature module.
    """
    rows = []
    for x in indicators:
        for y in indicators:
            if x != y and (e := fit_edge(panel, x, y, **kwargs)) is not None:
                rows.append(e.__dict__)
    out = pd.DataFrame(rows)
    z = stats.norm.ppf(1 - alpha / 2 / max(len(out), 1))
    out["discovered"] = out["t"].abs() > z
    out.attrs["z_threshold"] = z
    return out


def direction_test(panel: pd.DataFrame, a: str, b: str, lags: int = 2, **kwargs) -> dict:
    """Temporal-precedence item for the registry (M7.3).

    Strictly lagged within-entity regressions in both directions (dY_t on dX_{t-1..t-lags}
    and dY_{t-1..t-lags}, with the same fixed effects as `fit_edge`), joint Wald tests,
    and an event-study profile around the largest residual steps in each variable
    (local projections h = -3..4, pre-trend reported). Series are aligned with
    `configs/indicator_reference_periods.yaml` before testing.

    Returns {'result': forward | reverse | both | undecided, 'p_forward', 'p_reverse',
    'pre_trend_p', 'event_profile'}. The result is one weighted evidence item; it never
    overrides mechanism, literature or SD structure (M7 analysis: on annual NUTS-2 data
    6 of 24 prior edges forward, 4 reverse, 13 undecided).
    """
    raise NotImplementedError("M7.3")


def heterogeneity(panel: pd.DataFrame, cause: str, effect: str, groups: pd.Series, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Estimate an edge per entity group and test for heterogeneity (Cochran's Q, I²).

    `groups` maps entity id to a group label (e.g. a macro-region or a typology class from
    `regsim.typology`).
    """
    entity = kwargs.get("entity", "geo_id")
    rows = []
    for lab, ids in groups.groupby(groups).groups.items():
        e = fit_edge(panel[panel[entity].isin(ids)], cause, effect, **kwargs)
        if e is not None:
            rows.append({"group": lab, **e.__dict__})
    tab = pd.DataFrame(rows)
    se = tab["ci95_std"].map(lambda c: c[1] - c[0]) / 3.92
    w = 1 / se**2
    mu = float((w * tab["beta_std"]).sum() / w.sum())
    Q = float((w * (tab["beta_std"] - mu) ** 2).sum())
    k = len(tab) - 1
    return tab, {"Q": Q, "df": k, "p": float(1 - stats.chi2.cdf(Q, k)), "I2": max(0.0, (Q - k) / Q) if Q > 0 else 0.0}
