"""Panel effects of the registry edges (C2), pooled and per region group.

One row per edge and group (``"all"`` for the pooled estimate) with the cumulative short-run effect
and its long-run transform on the transformed scales of cause and effect (``t_cause``,
``t_effect``), group estimates partially pooled towards the pooled effect (decision D14). The table
is stored in the Parquet evidence store (decision D10) and read by the causal queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regsim.causal.groups import pooled_effects
from regsim.causal.panel import fit_edge


def transform_of(indicator: str, registry=None, transforms: dict[str, str] | None = None) -> str:
    """Transform of ``indicator`` in the panels: the causal override, else the registry, else identity."""
    if transforms and indicator in transforms:
        return transforms[indicator]
    if registry is not None and indicator in registry.entries:
        return str(registry[indicator].transform)
    return "identity"


def estimate_effects(
    edges,
    panel: pd.DataFrame,
    groups: pd.Series | None = None,
    registry=None,
    transforms: dict[str, str] | None = None,
    statuses: tuple[str, ...] = ("prior", "supported", "contested", "hypothesis"),
    **kwargs,
) -> pd.DataFrame:
    """Panel estimates for every edge with both ends in ``panel``.

    Parameters
    ----------
    edges : EdgeRegistry
    panel : DataFrame
        Annual panel from :func:`regsim.causal.data.annual_panel` (transformed).
    groups : Series, optional
        Entity id -> group (typology class or macro-region); adds partially pooled group rows.
    registry : IndicatorRegistry, optional
        Source of the transforms recorded with each row.
    transforms : dict, optional
        Causal transform overrides (``causal.transforms``).
    **kwargs
        Passed to :func:`regsim.causal.panel.fit_edge` (``method``, ``bias_correction``).
    """
    rows = []
    seen = set()
    for e in edges.edges:
        if e.status not in statuses or e.cause not in panel.columns or e.effect not in panel.columns:
            continue
        key = (e.cause, e.effect)
        common = {
            "cause": e.cause,
            "effect": e.effect,
            "t_cause": transform_of(e.cause, registry, transforms),
            "t_effect": transform_of(e.effect, registry, transforms),
        }
        if key in seen:  # several registry edges for one pair share the estimate
            rows += [
                {**r, "edge_id": e.id}
                for r in rows
                if (r["cause"], r["effect"]) == key and r["edge_id"] != e.id
            ]
            continue
        seen.add(key)
        est = fit_edge(panel, e.cause, e.effect, **kwargs)
        if est is None:
            continue
        rows.append(
            {
                "edge_id": e.id,
                **common,
                "group": "all",
                "beta_raw": est.beta_raw,
                "se_raw": est.se_raw,
                "beta_pooled": est.beta_raw,
                "se_pooled": est.se_raw,
                "long_run_raw": est.long_run_raw,
                "se_long_run_raw": est.se_long_run_raw,
                "rho": est.rho,
                "beta_std": est.beta_std,
                "t": est.t,
                "ci95_lo_std": est.ci95_std[0],
                "ci95_hi_std": est.ci95_std[1],
                "n_entities": est.n_entities,
                "method": est.method,
            }
        )
        if groups is None:
            continue
        tab, summary = pooled_effects(panel, e.cause, e.effect, groups, **kwargs)
        for r in tab.to_dict("records"):
            lr_factor = est.long_run_raw / est.beta_raw if est.beta_raw else np.nan
            rows.append(
                {
                    "edge_id": e.id,
                    **common,
                    "group": r["group"],
                    "beta_raw": r["beta_raw"],
                    "se_raw": r["se_raw"],
                    "beta_pooled": r["beta_pooled"],
                    "se_pooled": r["se_pooled"],
                    "long_run_raw": r["long_run_raw"],
                    "se_long_run_raw": r["se_pooled"] * abs(lr_factor) if np.isfinite(lr_factor) else np.nan,
                    "rho": np.nan,
                    "beta_std": r["beta_std"],
                    "t": r["t"],
                    "ci95_lo_std": float("nan"),
                    "ci95_hi_std": float("nan"),
                    "n_entities": r["n_entities"],
                    "method": est.method,
                    "I2": summary["I2"],
                    "p_heterogeneity": summary["p"],
                }
            )
    return pd.DataFrame(rows)
