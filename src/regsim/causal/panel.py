"""Within-region panel estimation of causal edges (option C2).

Reference implementation from the M7 analysis (causal_test_notes.md). The model for an
edge X -> Y on a long panel of entity-years is

    dY_t = b0 dX_t + b1 dX_{t-1} + rho dY_{t-1} + a_g + d_{c,t} + e_gt

with entity effects a_g and parent-by-year effects d_{c,t} (country x year for NUTS-2;
year effects for national panels). The reported effect is the cumulative short-run
effect b0 + b1, standardised by the demeaned standard deviations of dX and dY, with
cluster-robust standard errors by entity. The long-run effect is (b0 + b1) / (1 - rho).

With short panels the within estimator of rho is biased (Nickell bias, about -(1 + rho) / (T - 1)).
Decision D13: the default removes it with the half-panel jackknife; Arellano-Bond difference
GMM is an option.

Parent-by-year groups come from the hierarchy graph of the `GeoScheme`, never from code
prefixes (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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
    method: str = "fe"


def demean(
    df: pd.DataFrame, keys: list[str], cols: list[str], tol: float = 1e-10, max_iter: int = 200
) -> pd.DataFrame:
    """Remove several sets of fixed effects by alternating projections."""
    X = df[cols].astype(float).copy()
    for _ in range(max_iter):
        X_old = X.copy()
        for k in keys:
            X = X - X.groupby(df[k]).transform("mean")
        if float(np.nanmax(np.abs((X - X_old).to_numpy()))) < tol:
            break
    return X


def _prepare(
    panel: pd.DataFrame, cause: str, effect: str, entity: str, time: str, group: str | None
) -> pd.DataFrame:
    d = panel[[entity, time, cause, effect] + ([group] if group else [])].sort_values([entity, time]).copy()
    g = d.groupby(entity)
    d["dy"] = g[effect].diff()
    d["dx0"] = g[cause].diff()
    d["dx1"] = d.groupby(entity)["dx0"].shift(1)
    d["dy_l"] = d.groupby(entity)["dy"].shift(1)
    d["dy_l2"] = d.groupby(entity)["dy"].shift(2)  # instruments for the GMM option
    d["dy_l3"] = d.groupby(entity)["dy"].shift(3)
    d = d.dropna(subset=["dy", "dx0", "dx1", "dy_l"]).copy()
    d["fe_time"] = d[group].astype(str) + "_" + d[time].astype(str) if group else d[time].astype(str)
    return d


def _cluster_meat(X: np.ndarray, res: np.ndarray, clusters) -> np.ndarray:
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in clusters:
        s = X[idx].T @ res[idx]
        meat += np.outer(s, s)
    return meat


def _fe(d: pd.DataFrame, entity: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Within estimator with entity and parent-by-year effects; cluster-robust covariance by entity."""
    Xd = demean(d, [entity, "fe_time"], ["dy", "dx0", "dx1", "dy_l"])
    X = Xd[["dx0", "dx1", "dy_l"]].to_numpy()
    y = Xd["dy"].to_numpy()
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    G = d[entity].nunique()
    V = XtX @ _cluster_meat(X, y - X @ b, d.groupby(entity).indices.values()) @ XtX * G / (G - 1)
    return b, V, float(Xd["dx0"].std()), float(Xd["dy"].std())


def _jackknife(d: pd.DataFrame, entity: str, time: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Half-panel jackknife (Dhaene and Jochmans 2015): 2 b_full - (b_first + b_second) / 2 removes the
    O(1/T) Nickell bias of the within estimator; the covariance is that of the full panel."""
    b, V, sx, sy = _fe(d, entity)
    years = np.sort(d[time].unique())
    cut = years[len(years) // 2]
    halves = [d[d[time] < cut], d[d[time] >= cut]]
    if min(h[entity].nunique() for h in halves) < 2 or min(len(h) for h in halves) < 10:
        return b, V, sx, sy
    b_half = [_fe(h, entity)[0] for h in halves]
    return 2 * b - (b_half[0] + b_half[1]) / 2, V, sx, sy


def _gmm(d: pd.DataFrame, entity: str, time: str) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Arellano-Bond difference GMM, one step, collapsed instruments.

    Parent-by-year effects are removed by demeaning within parent-year cells; the entity effects by
    first-differencing the equation. The differenced lagged dependent variable is instrumented with
    dY_{t-2} and dY_{t-3} (one column per lag depth, zero where missing); the cause terms are
    treated as exogenous.
    """
    cols = ["dy", "dx0", "dx1", "dy_l", "dy_l2", "dy_l3"]
    e = d[[entity, time, "fe_time"]].copy()
    e[cols] = d[cols] - d[cols].groupby(d["fe_time"]).transform("mean")
    e = e.sort_values([entity, time])
    g = e.groupby(entity)
    diff = g[["dy", "dx0", "dx1", "dy_l"]].diff()
    consecutive = g[time].diff() == 1
    ok = consecutive & diff.notna().all(axis=1)
    X = diff.loc[ok, ["dx0", "dx1", "dy_l"]].to_numpy()
    y = diff.loc[ok, "dy"].to_numpy()
    Z = np.column_stack([X[:, 0], X[:, 1], e.loc[ok, ["dy_l2", "dy_l3"]].fillna(0.0).to_numpy()])
    ent = e.loc[ok, entity].to_numpy()
    W = np.linalg.pinv(Z.T @ Z)
    XZ = X.T @ Z
    A = np.linalg.pinv(XZ @ W @ XZ.T)
    b = A @ XZ @ W @ Z.T @ y
    clusters = pd.Series(np.arange(len(ent))).groupby(ent).indices.values()
    S = _cluster_meat(Z, y - X @ b, clusters)
    G = len(set(ent))
    V = A @ XZ @ W @ S @ W @ XZ.T @ A * G / (G - 1)
    return b, V, float(np.std(X[:, 0])), float(np.std(y)), int(ok.sum())


def fit_edge(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
    min_obs: int = 60,
    min_entities: int = 12,
    method: str = "fe",
    bias_correction: str = "jackknife",
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
    method
        ``"fe"``: within estimator (decision D13 default); ``"gmm"``: Arellano-Bond difference GMM.
    bias_correction
        For ``"fe"``: ``"jackknife"`` (half-panel jackknife against the Nickell bias, D13) or
        ``"none"`` (plain within estimator, the M7 reference values).

    Returns
    -------
    EdgeEstimate or None if the panel is too small.
    """
    d = _prepare(panel, cause, effect, entity, time, group)
    if len(d) < min_obs or d[entity].nunique() < min_entities:
        return None
    n_obs = len(d)
    if method == "gmm":
        b, V, sx, sy, n_obs = _gmm(d, entity, time)
        label = "gmm"
    elif bias_correction == "jackknife":
        b, V, sx, sy = _jackknife(d, entity, time)
        label = "fe_jackknife"
    elif bias_correction == "none":
        b, V, sx, sy = _fe(d, entity)
        label = "fe"
    else:
        raise ValueError(f"unknown method {method!r} / bias_correction {bias_correction!r}")
    c = np.array([1.0, 1.0, 0.0])
    cum, se = float(c @ b), float(np.sqrt(c @ V @ c))
    rho = float(b[2])
    grad = np.array([1 / (1 - rho), 1 / (1 - rho), cum / (1 - rho) ** 2])
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
        n_obs=n_obs,
        n_entities=d[entity].nunique(),
        method=label,
    )


def screen(
    panel: pd.DataFrame,
    indicators: list[str],
    alpha: float = 0.05,
    registry_indicators=None,
    **kwargs,
) -> pd.DataFrame:
    """Fit all ordered pairs and flag those above a Bonferroni threshold.

    Pairs flagged here are candidates only. With ``registry_indicators`` each pair gets its
    ``mechanism`` from `graph.classify_mechanism`, and ``to_literature`` marks the discovered pairs
    that are not accounting identities, shared denominators or shared survey bases; only these go to
    the literature module.
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
    if registry_indicators is not None and len(out):
        from regsim.causal.graph import classify_mechanism

        out["mechanism"] = [
            classify_mechanism(c, e, registry_indicators) for c, e in zip(out.cause, out.effect, strict=True)
        ]
        out["to_literature"] = out.discovered & (out.mechanism == "causal")
    return out


def reference_offsets(path: str | Path) -> dict[str, int]:
    """Year offset of each indicator's reference period from ``configs/indicator_reference_periods.yaml``.

    -1 for measures whose reference period is mostly the previous year (EU-SILC income: survey year t,
    income year t-1), 0 otherwise. Stocks on 1 January keep the year of the date, as in the M7
    analysis.
    """
    text = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["indicators"]
    return {k: -1 if "mostly t-1" in str(v).lower() else 0 for k, v in text.items()}


def align(panel: pd.DataFrame, offsets: dict[str, int], time: str = "year") -> pd.DataFrame:
    """Re-index each indicator by its reference year (``year + offset``)."""
    keys = [c for c in ("geo_id", "parent_id") if c in panel.columns]
    out = panel[[*keys, time]].copy()
    for c in panel.columns:
        if c in (*keys, time):
            continue
        shift = offsets.get(c, 0)
        col = panel[[keys[0], time, c]].assign(**{time: panel[time] + shift})
        out = out.drop(columns=c, errors="ignore").merge(col, on=[keys[0], time], how="left")
    return out


def _lagged(
    panel: pd.DataFrame, x: str, y: str, lags: int, entity: str, time: str, group: str | None
) -> dict:
    """dY_t on dX_{t-1..t-lags} and dY_{t-1..t-lags} with entity and parent-by-year effects; joint
    cluster-robust Wald test of the dX lags."""
    d = panel[[entity, time, x, y] + ([group] if group else [])].sort_values([entity, time]).copy()
    g = d.groupby(entity)
    d["dy"] = g[y].diff()
    d["dx"] = g[x].diff()
    names = []
    for k in range(1, lags + 1):
        d[f"dx_l{k}"] = d.groupby(entity)["dx"].shift(k)
        d[f"dy_l{k}"] = d.groupby(entity)["dy"].shift(k)
        names += [f"dx_l{k}", f"dy_l{k}"]
    d = d.dropna(subset=["dy", *names]).copy()
    d["fe_time"] = d[group].astype(str) + "_" + d[time].astype(str) if group else d[time].astype(str)
    if len(d) < 60 or d[entity].nunique() < 12:
        return {"p": np.nan, "wald": np.nan, "cum": np.nan, "t_cum": np.nan, "n": len(d)}
    Xd = demean(d, [entity, "fe_time"], ["dy", *names])
    X = Xd[names].to_numpy()
    yv = Xd["dy"].to_numpy()
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ yv
    G = d[entity].nunique()
    V = XtX @ _cluster_meat(X, yv - X @ b, d.groupby(entity).indices.values()) @ XtX * G / (G - 1)
    ix = [names.index(f"dx_l{k}") for k in range(1, lags + 1)]
    bx, Vx = b[ix], V[np.ix_(ix, ix)]
    wald = float(bx @ np.linalg.pinv(Vx) @ bx)
    c = np.zeros(len(names))
    c[ix] = 1.0
    cum, se = float(c @ b), float(np.sqrt(c @ V @ c))
    if not se > 0:  # degenerate pair (e.g. a series without within-entity variation)
        return {"p": np.nan, "wald": np.nan, "cum": cum, "t_cum": np.nan, "n": len(d)}
    return {
        "p": float(1 - stats.chi2.cdf(wald, lags)),
        "wald": wald,
        "cum": cum,
        "t_cum": cum / se,
        "n": len(d),
    }


def event_profile(
    panel: pd.DataFrame,
    step_in: str,
    response: str,
    horizons: range = range(-3, 5),
    quantile: float = 0.9,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
) -> tuple[pd.DataFrame, float]:
    """Local projections of ``response`` around large steps in ``step_in``.

    Events are entity-years whose change in ``step_in`` (after removing entity and parent-by-year
    means) lies above the ``quantile`` of absolute changes, signed by the direction of the step. For
    each horizon h, y_{t+h} - y_{t-1} is regressed on the signed event with parent-by-year effects.

    Returns
    -------
    profile : DataFrame
        ``h``, ``beta``, ``se``, ``n_events``.
    pre_trend_p : float
        Joint test that the responses before the step (h < -1) are zero.
    """
    d = (
        panel[[entity, time, step_in, response] + ([group] if group else [])]
        .sort_values([entity, time])
        .copy()
    )
    d["dx"] = d.groupby(entity)[step_in].diff()
    d["fe_time"] = d[group].astype(str) + "_" + d[time].astype(str) if group else d[time].astype(str)
    ok = d.dx.notna()
    z = pd.Series(np.nan, index=d.index)
    z[ok] = demean(d[ok], [entity, "fe_time"], ["dx"])["dx"]
    cut = z.abs().quantile(quantile)
    d["event"] = np.where(z.abs() >= cut, np.sign(z), 0.0)
    d.loc[z.isna(), "event"] = np.nan
    base = d.groupby(entity)[response].shift(1)
    rows, pre = [], []
    for h in horizons:
        if h == -1:
            rows.append({"h": h, "beta": 0.0, "se": 0.0, "n_events": int((d.event != 0).sum())})
            continue
        e = d.assign(dyh=d.groupby(entity)[response].shift(-h) - base).dropna(subset=["dyh", "event"])
        Xd = demean(e, ["fe_time"], ["dyh", "event"])
        x, yv = Xd["event"].to_numpy(), Xd["dyh"].to_numpy()
        beta = float(x @ yv / (x @ x))
        res = yv - beta * x
        meat = sum(float(x[i] @ res[i]) ** 2 for i in e.groupby(entity).indices.values())
        se = float(np.sqrt(meat) / (x @ x))
        rows.append({"h": h, "beta": beta, "se": se, "n_events": int((e.event != 0).sum())})
        if h < -1:
            pre.append((beta, se))
    wald = sum((b / s) ** 2 for b, s in pre if s > 0)
    return pd.DataFrame(rows), float(1 - stats.chi2.cdf(wald, len(pre))) if pre else np.nan


def direction_test(
    panel: pd.DataFrame,
    a: str,
    b: str,
    lags: int = 2,
    alpha: float = 0.05,
    n_tests: int = 1,
    offsets: dict[str, int] | None = None,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
    events: bool = True,
) -> dict:
    """Temporal-precedence item for the registry (M7.3).

    Strictly lagged within-entity regressions in both directions (dY_t on dX_{t-1..t-lags}
    and dY_{t-1..t-lags}, with the same fixed effects as `fit_edge`), joint Wald tests,
    and an event-study profile around the largest residual steps in each variable
    (local projections h = -3..4, pre-trend reported). Series are aligned with
    `configs/indicator_reference_periods.yaml` (``offsets`` from `reference_offsets`) before testing.

    A direction counts as significant at ``alpha / n_tests`` (Bonferroni over all tests of a batch,
    two per pair; the M7 analysis used 240 tests for 120 pairs).

    Returns {'result': forward | reverse | both | undecided, 'p_forward', 'p_reverse',
    'pre_trend_p', 'event_profile'}. The result is one weighted evidence item; it never
    overrides mechanism, literature or SD structure (M7 analysis: on annual NUTS-2 data
    6 of 24 prior edges forward, 4 reverse, 13 undecided).
    """
    if offsets:
        panel = align(panel, offsets, time)
    fwd = _lagged(panel, a, b, lags, entity, time, group)
    rev = _lagged(panel, b, a, lags, entity, time, group)
    thr = alpha / n_tests
    f, r = fwd["p"] < thr, rev["p"] < thr
    result = {(True, False): "forward", (False, True): "reverse", (True, True): "both"}.get(
        (f, r), "undecided"
    )
    out = {
        "result": result,
        "p_forward": fwd["p"],
        "p_reverse": rev["p"],
        "beta_forward": fwd["cum"],
        "beta_reverse": rev["cum"],
        "threshold": thr,
        "lags": lags,
        "aligned": bool(offsets),
        "pre_trend_p": np.nan,
        "event_profile": None,
    }
    if events:
        prof, pre_p = event_profile(panel, a, b, entity=entity, time=time, group=group)
        out.update(pre_trend_p=pre_p, event_profile=prof.to_dict("records"))
    return out


def heterogeneity(
    panel: pd.DataFrame, cause: str, effect: str, groups: pd.Series, **kwargs
) -> tuple[pd.DataFrame, dict]:
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
    return tab, {
        "Q": Q,
        "df": k,
        "p": float(1 - stats.chi2.cdf(Q, k)),
        "I2": max(0.0, (Q - k) / Q) if Q > 0 else 0.0,
    }
