"""Region groups for partial pooling of edge effects and for moderators (decision D14).

Typology classes from `regsim.typology` are the default grouping; macro-regions from the config
(``causal.macro_regions``: group -> country codes) are the fallback. Countries are found through
the ``parent_id`` column of the panel (level-0 ancestor from the hierarchy graph).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regsim.api.runtime import short_id
from regsim.causal.panel import fit_edge


def macro_groups(parents: pd.Series, macro_regions: dict[str, list[str]]) -> pd.Series:
    """Entity id -> macro-region, from a series entity id -> country id; entities of unlisted
    countries are left out."""
    code_to_group = {c: g for g, codes in macro_regions.items() for c in codes}
    out = parents.map(lambda p: code_to_group.get(short_id(p)) if isinstance(p, str) else None)
    return out.dropna()


def typology_groups(fitted, level: int, k: int | None = None, seed: int = 0, n_boot: int = 10) -> pd.Series:
    """Entity id -> typology class (``"type <i>"``) of the entities at ``level`` (GMM on the whitened
    profiles, k chosen by stability and BIC unless given)."""
    from regsim.schema import Lens
    from regsim.typology.run import TypologyRun, run_typology

    ty = run_typology(
        fitted.base, fitted.data, TypologyRun(lens=Lens(levels=(level,)), k=k, n_boot=n_boot, seed=seed)
    )
    return pd.Series([f"type {t}" for t in ty.labels], index=pd.Index(ty.geo_ids, name="geo_id"))


def region_groups(cfg, panel: pd.DataFrame, fitted=None) -> pd.Series:
    """Groups for partial pooling as configured (decision D14): typology classes when ``causal.groups``
    is ``"typology"`` and a fitted model is given, else macro-regions."""
    c = cfg.causal
    if c is not None and c.groups == "typology" and fitted is not None:
        return typology_groups(fitted, c.level, seed=cfg.model.seed)
    parents = panel.drop_duplicates("geo_id").set_index("geo_id")["parent_id"]
    return macro_groups(parents, c.macro_regions if c is not None else {})


def pooled_effects(
    panel: pd.DataFrame, cause: str, effect: str, groups: pd.Series, min_entities: int = 8, **kwargs
) -> tuple[pd.DataFrame, dict]:
    """Edge effect per group with partial pooling towards the pooled effect.

    The cumulative raw effect of each group is shrunk towards the precision-weighted mean with the
    between-group variance tau^2 from the DerSimonian-Laird estimator: precise groups keep their
    estimate, noisy ones move towards the mean.

    Returns
    -------
    table : DataFrame
        One row per group: ``group``, ``beta_raw``, ``se_raw``, ``beta_pooled`` (shrunk), ``se_pooled``,
        ``beta_std``, ``t``, ``n_entities``.
    summary : dict
        ``mu`` (pooled raw effect), ``tau2``, ``Q``, ``I2``, ``p``.
    """
    from scipy import stats

    rows = []
    for lab, ids in groups.groupby(groups).groups.items():
        e = fit_edge(panel[panel["geo_id"].isin(ids)], cause, effect, min_entities=min_entities, **kwargs)
        if e is not None:
            rows.append(
                {
                    "group": lab,
                    "beta_raw": e.beta_raw,
                    "se_raw": e.se_raw,
                    "beta_std": e.beta_std,
                    "t": e.t,
                    "long_run_raw": e.long_run_raw,
                    "n_entities": e.n_entities,
                }
            )
    tab = pd.DataFrame(rows)
    if len(tab) < 2:
        return tab, {"mu": np.nan, "tau2": np.nan, "Q": np.nan, "I2": np.nan, "p": np.nan}
    w = 1 / tab.se_raw**2
    mu_fe = float((w * tab.beta_raw).sum() / w.sum())
    Q = float((w * (tab.beta_raw - mu_fe) ** 2).sum())
    k = len(tab) - 1
    tau2 = max(0.0, (Q - k) / float(w.sum() - (w**2).sum() / w.sum()))
    w_re = 1 / (tab.se_raw**2 + tau2)
    mu = float((w_re * tab.beta_raw).sum() / w_re.sum())
    if tau2 > 0:
        prec = 1 / tab.se_raw**2 + 1 / tau2
        tab["beta_pooled"] = (tab.beta_raw / tab.se_raw**2 + mu / tau2) / prec
        tab["se_pooled"] = np.sqrt(1 / prec)
    else:
        tab["beta_pooled"] = mu
        tab["se_pooled"] = float(np.sqrt(1 / w.sum()))
    return tab, {
        "mu": mu,
        "tau2": tau2,
        "Q": Q,
        "I2": max(0.0, (Q - k) / Q) if Q > 0 else 0.0,
        "p": float(1 - stats.chi2.cdf(Q, k)),
    }
