"""Effect similarity: region-specific causal effects as derived indicators (M7.6).

Method (docs/causal/effect_similarity.md):
- `fit_varying`: within-region panel model of an edge (as `causal.panel.fit_edge`) in which
  the cumulative effect varies linearly with standardised moderators (mean log GDP, mean
  level of the effect indicator, mean log density, region type). Moderator terms carry a
  ridge penalty chosen by leave-one-country-out validation.
- Validation gate: a link is heterogeneous only if moderators reduce the hold-out error by
  at least 10 % against one pooled effect (M7 analysis: GDP -> life expectancy 0.37,
  GDP -> unemployment 0.73, GDP -> cars 0.77; four other links failed the gate).
- `region_effects`: predicted effect, interval and `heterogeneous` flag per entity and link.
  Stored in the observation store as derived indicators `effect:<cause>-><effect>`
  (scenario `effect`, horizon short run). Not used as inputs to HierTypedMF, which reads the
  scenario `observed` only.
- `s_effect`: effect distance and probability of practical equivalence
  P(|beta_A - beta_B| < delta) between two entities, using the joint covariance of
  the predictions; delta = 0.1 SD by default, set per link (decision D19).

Moderator sets and region types come from configuration and `regsim.typology`, not
from geo-code prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from regsim.causal.panel import _cluster_meat, _prepare, demean, fit_edge

ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)


def link_id(cause: str, effect: str) -> str:
    return f"{cause}->{effect}"


def moderators(
    panel: pd.DataFrame, effect: str, groups: pd.Series | None = None, extra: list[str] | None = None
) -> pd.DataFrame:
    """Standardised moderators per entity: mean of GDP per inhabitant, of the effect indicator and of
    density (transformed scales of the panel), and dummies of the region groups (largest group as
    reference)."""
    cols = [c for c in dict.fromkeys(["gdp_pps_hab", effect, "pop_density", *(extra or [])]) if c in panel]
    m = panel.groupby("geo_id")[cols].mean()
    m.columns = [f"mean_{c}" for c in cols]
    m = (m - m.mean()) / m.std()
    if groups is not None:
        g = groups.reindex(m.index)
        ref = g.value_counts().idxmax()
        for lab in sorted(g.dropna().unique()):
            if lab != ref:
                m[f"group_{lab}"] = (g == lab).astype(float)
    return m.fillna(0.0)


@dataclass
class VaryingFit:
    """Varying-coefficient fit of one link."""

    cause: str
    effect: str
    alpha: float
    coef: np.ndarray  # [b0, b1, rho, gamma0 (moderators), gamma1 (moderators)]
    cov: np.ndarray
    moderators: pd.DataFrame  # standardised moderators of all entities
    sx: float
    sy: float
    n_obs: int
    heterogeneous: bool = True
    gate: dict = field(default_factory=dict)
    pooled: object = None  # EdgeEstimate of the pooled effect

    @property
    def link(self) -> str:
        return link_id(self.cause, self.effect)

    def contrast(self, geo_id: str) -> np.ndarray:
        """Vector c with c @ coef = cumulative raw effect of ``geo_id``."""
        m = (
            self.moderators.loc[geo_id].to_numpy()
            if geo_id in self.moderators.index
            else np.zeros(self.moderators.shape[1])
        )
        if not self.heterogeneous:
            m = np.zeros_like(m)
        return np.concatenate([[1.0, 1.0, 0.0], m, m])

    def effect_of(self, geo_ids) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative raw effects and their covariance for ``geo_ids``."""
        if not self.heterogeneous and self.pooled is not None:
            n = len(geo_ids)
            return np.full(n, self.pooled.beta_raw), np.full((n, n), self.pooled.se_raw**2)
        C = np.array([self.contrast(g) for g in geo_ids])
        return C @ self.coef, C @ self.cov @ C.T


def _design(d: pd.DataFrame, mods: pd.DataFrame, entity: str) -> tuple[pd.DataFrame, list[str], int]:
    m = mods.reindex(d[entity]).to_numpy()
    k = m.shape[1]
    X = d[["dy", "dx0", "dx1", "dy_l"]].copy()
    names = ["dx0", "dx1", "dy_l"]
    for j in range(k):
        X[f"m0_{j}"] = d["dx0"].to_numpy() * m[:, j]
        names.append(f"m0_{j}")
    for j in range(k):
        X[f"m1_{j}"] = d["dx1"].to_numpy() * m[:, j]
        names.append(f"m1_{j}")
    return X, names, k


def fit_varying(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    moderators: pd.DataFrame,
    alpha: float | None = None,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
    countries: pd.Series | None = None,
    **kwargs,
) -> VaryingFit | None:
    """Fit the varying-coefficient model; ``alpha=None`` selects the penalty by leave-one-country-out
    validation (`validate_gate`). The penalty is relative: ``alpha`` times the diagonal of X'X of the
    moderator terms, so that it does not depend on their scale; ``alpha=inf`` is one pooled effect."""
    if alpha is None:
        countries = (
            countries if countries is not None else panel.drop_duplicates(entity).set_index(entity)[group]
        )
        gate = validate_gate(
            panel, cause, effect, moderators, countries, entity=entity, time=time, group=group
        )
        fit = fit_varying(panel, cause, effect, moderators, gate["alpha"], entity, time, group)
        if fit is not None:
            fit.gate, fit.heterogeneous = gate, gate["heterogeneous"]
        return fit
    d = _prepare(panel, cause, effect, entity, time, group)
    d = d[d[entity].isin(moderators.index)]
    if len(d) < 60 or d[entity].nunique() < 12:
        return None
    X, names, k = _design(d, moderators, entity)
    Xd = demean(d.assign(**{c: X[c] for c in X.columns}), [entity, "fe_time"], ["dy", *names])
    A = Xd[names].to_numpy()
    y = Xd["dy"].to_numpy()
    pen = np.zeros(len(names))
    G = A.T @ A
    if np.isinf(alpha):  # pooled: moderator terms fixed at zero
        keep = np.arange(3)
        b = np.zeros(len(names))
        Ginv = np.linalg.pinv(G[np.ix_(keep, keep)])
        b[keep] = Ginv @ A[:, keep].T @ y
        H = np.zeros_like(G)
        H[np.ix_(keep, keep)] = Ginv
    else:
        pen[3:] = alpha * np.diag(G)[3:]
        H = np.linalg.pinv(G + np.diag(pen))
        b = H @ A.T @ y
    meat = _cluster_meat(A, y - A @ b, d.groupby(entity).indices.values())
    n_g = d[entity].nunique()
    V = H @ meat @ H * n_g / (n_g - 1)
    pooled = fit_edge(
        panel[panel[entity].isin(moderators.index)], cause, effect, entity, time, group, **kwargs
    )
    return VaryingFit(
        cause=cause,
        effect=effect,
        alpha=float(alpha),
        coef=b,
        cov=V,
        moderators=moderators,
        sx=float(Xd["dx0"].std()),
        sy=float(Xd["dy"].std()),
        n_obs=len(d),
        heterogeneous=not np.isinf(alpha),
        pooled=pooled,
    )


def validate_gate(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    moderators: pd.DataFrame,
    groups: pd.Series,
    threshold: float = 0.9,
    alphas: tuple[float, ...] = ALPHAS,
    min_regions: int = 3,
    entity: str = "geo_id",
    time: str = "year",
    group: str | None = "parent_id",
) -> dict:
    """Leave-one-group-out comparison with the pooled effect; returns relative error and decision.

    For each group (country) with at least ``min_regions`` entities, the model is fitted without it and
    the group's mean predicted effect is compared with the group's own estimate (`fit_edge` on its
    data alone, year effects). Squared errors are weighted by the precision of the own estimate. The
    relative error is the error of the varying model divided by that of a pooled effect fitted on the
    same entities. The penalty with the smallest relative error is chosen; the link is heterogeneous if
    it is at most ``threshold``.

    Returns
    -------
    dict
        ``alpha``, ``rel_err`` (at the chosen penalty), ``rel_err_unshrunk`` (alpha = 0),
        ``n_groups``, ``heterogeneous``, ``per_group`` (DataFrame).
    """
    rows = []
    for lab, ids in groups.groupby(groups).groups.items():
        ids = [i for i in ids if i in moderators.index]
        if len(ids) < min_regions:
            continue
        own = fit_edge(
            panel[panel[entity].isin(ids)], cause, effect, entity, time, None, min_obs=20, min_entities=min_regions,
            bias_correction="none",
        )  # fmt: skip
        if own is None or not own.se_raw > 0:
            continue
        rest = panel[~panel[entity].isin(ids)]
        pool = fit_edge(rest, cause, effect, entity, time, group, bias_correction="none")
        if pool is None:
            continue
        row = {"group": lab, "own": own.beta_raw, "se": own.se_raw, "pred_pooled": pool.beta_raw}
        for a in alphas:
            if np.isinf(a):
                row[f"pred_{a}"] = pool.beta_raw
                continue
            f = fit_varying(
                rest, cause, effect, moderators.drop(index=ids, errors="ignore"), a, entity, time, group
            )
            if f is None:
                row[f"pred_{a}"] = np.nan
                continue
            f.moderators = moderators  # predict the held-out entities with their own moderators
            row[f"pred_{a}"] = float(np.mean(f.effect_of(ids)[0]))
        rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        return {"alpha": np.inf, "rel_err": np.nan, "rel_err_unshrunk": np.nan, "n_groups": 0,
                "heterogeneous": False, "per_group": tab}  # fmt: skip
    w = 1.0 / tab.se**2

    def loss(pred: pd.Series) -> float:
        ok = pred.notna()
        return float((w[ok] * (tab.own[ok] - pred[ok]) ** 2).sum() / w[ok].sum())

    base = loss(tab.pred_pooled)
    rel = {a: loss(tab[f"pred_{a}"]) / base for a in alphas}
    best = min(rel, key=rel.get)
    return {
        "alpha": float(best),
        "rel_err": float(rel[best]),
        "rel_err_unshrunk": float(rel.get(0.0, np.nan)),
        "rel_err_by_alpha": {float(a): float(v) for a, v in rel.items()},
        "n_groups": len(tab),
        "heterogeneous": bool(rel[best] <= threshold and not np.isinf(best)),
        "per_group": tab,
    }


def region_effects(fit: VaryingFit, entities: pd.Index, level: float = 0.9) -> pd.DataFrame:
    """Predicted cumulative effect per entity: raw and standardised (by the within-entity standard
    deviations of the changes), ``level`` interval, ``heterogeneous`` flag (pooled effect otherwise)."""
    ids = list(entities)
    mu, cov = fit.effect_of(ids)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    s = fit.sx / fit.sy
    return pd.DataFrame(
        {
            "geo_id": ids,
            "link": fit.link,
            "cause": fit.cause,
            "effect": fit.effect,
            "heterogeneous": fit.heterogeneous,
            "alpha": fit.alpha,
            "beta_raw": mu,
            "se_raw": se,
            "beta_std": mu * s,
            "lo_std": (mu - z * se) * s,
            "hi_std": (mu + z * se) * s,
            "horizon": "short_run",
        }
    )


def equivalence_probability(diff: float, se: float, delta: float) -> float:
    """P(|true difference| < delta) for a difference estimate ``diff`` with standard error ``se``."""
    if not se > 0:
        return float(abs(diff) < delta)
    return float(stats.norm.cdf((delta - diff) / se) - stats.norm.cdf((-delta - diff) / se))


def s_effect(fits: dict, geo_a: str, geo_b: str, delta: float | dict[str, float] = 0.1) -> pd.DataFrame:
    """Per-link difference, standard error and probability of practical equivalence.

    Parameters
    ----------
    fits : dict
        Link id -> `VaryingFit`.
    delta : float or dict
        Practical-equivalence threshold in standard deviations of the effect indicator's change
        (decision D19: 0.1 by default, per link as a dict).

    Returns
    -------
    DataFrame
        One row per link: standardised effects of A and B with their standard errors, difference, its
        standard error,
        ``p_equivalent``; links without transferable heterogeneity have ``status`` "no transferable
        difference" and no probability. ``attrs['effect_distance']`` is the root mean square of the
        standardised differences over the heterogeneous links.
    """
    rows = []
    for link, f in fits.items():
        d_link = delta.get(link, 0.1) if isinstance(delta, dict) else delta
        s = f.sx / f.sy
        mu, cov = f.effect_of([geo_a, geo_b])
        diff = float(mu[0] - mu[1]) * s
        se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))) * s
        het = f.heterogeneous
        rows.append(
            {
                "link": link,
                "effect_a": float(mu[0]) * s,
                "effect_b": float(mu[1]) * s,
                "se_a": float(np.sqrt(max(cov[0, 0], 0.0))) * s,
                "se_b": float(np.sqrt(max(cov[1, 1], 0.0))) * s,
                "difference": diff if het else 0.0,
                "se": se if het else np.nan,
                "delta": d_link,
                "p_equivalent": equivalence_probability(diff, se, d_link) if het else np.nan,
                "status": "heterogeneous" if het else "no transferable difference",
            }
        )
    out = pd.DataFrame(rows)
    het = out[out.status == "heterogeneous"]
    out.attrs.update(
        geo_a=geo_a,
        geo_b=geo_b,
        effect_distance=float(np.sqrt((het.difference**2).mean())) if len(het) else np.nan,
    )
    return out


def effect_distances(fits: dict, geo_ids: list[str]) -> pd.DataFrame:
    """Effect distance between all pairs of ``geo_ids`` (root mean square of the standardised effect
    differences over the heterogeneous links)."""
    E = []
    for f in fits.values():
        if f.heterogeneous:
            E.append(f.effect_of(geo_ids)[0] * f.sx / f.sy)
    if not E:
        return pd.DataFrame(np.zeros((len(geo_ids), len(geo_ids))), index=geo_ids, columns=geo_ids)
    E = np.array(E).T
    D = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).mean(axis=2))
    return pd.DataFrame(D, index=geo_ids, columns=geo_ids)


def effect_observations(effects: pd.DataFrame, period: str, version: str) -> pd.DataFrame:
    """Region effects as derived observations ``effect:<cause>-><effect>`` (scenario ``effect``) for the
    observation store; value = standardised effect, ``se`` its standard error."""
    t = effects
    se = (t.hi_std - t.lo_std) / (2 * stats.norm.ppf(0.95))
    return pd.DataFrame(
        {
            "geo_id": t.geo_id,
            "indicator_id": "effect:" + t.link,
            "period": period,
            "value": t.beta_std,
            "scenario": "effect",
            "se": se,
            "source": "regsim:effect",
            "source_version": version,
            "flags": np.where(t.heterogeneous, "heterogeneous", "pooled") + ";" + t.horizon,
        }
    )
