"""Causal queries P8-P10.

Every result row carries `horizon` (short_run, long_run) and `tier` (E0-E4), and lists
the edges and sources used. Effects are propagated along registry edges with status
`prior` or `supported` and mechanism `causal`; contested edges are reported, not used.

Effect source per edge, in order of preference (open decision D12):
1. region-type-specific panel estimate (short run) or its long-run transform;
2. pooled panel estimate;
3. literature elasticity (long run);
4. sign only (direction without magnitude).
"""

from __future__ import annotations

import pandas as pd


def what_if(model, registry, geo_id: str, shocks: dict[str, float], horizon: str = "short_run", alpha: float = 0.1) -> pd.DataFrame:
    """P8: consequences of relative changes in `shocks` for the descendants of the shocked nodes.

    M7 example: +10 % GDP in EE00 -> cars per 1,000 +0.6 % (pooled short run), +1.1 %
    (eastern regions), +1.2 % (eastern long run), +3.0 % (cross-section), +4.6 %
    (literature).
    """
    raise NotImplementedError("M7.3")


def explain_difference(model, registry, geo_a: str, geo_b: str, target: str) -> pd.DataFrame:
    """P9: contributions of the target's parents to the difference between two entities,
    plus residual. A residual larger than the observed difference is reported as
    'missing causes'."""
    raise NotImplementedError("M7.3")


def vision_check(model, registry, geo_id: str, targets: dict[str, float], year: int, levers: list[str] | None = None) -> pd.DataFrame:
    """P10: required change in levers for each target given expected driver changes;
    flags conflicts and mismatched target variables (e.g. lever elasticity on car-km
    used for a car-ownership target). Extends `predict.tasks.vision_entity`."""
    raise NotImplementedError("M7.3")
