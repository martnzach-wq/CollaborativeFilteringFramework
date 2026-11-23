"""Causal-lens similarity (s_causal).

Similarity between entities restricted to the causes and effect modifiers of a target
indicator, taken from the edge registry. Optionally weighted by `TransferWeights`
learned for the target (`similarity.transfer`). Used by `predict.tasks.analogues`,
`causal.events.effect` and `causal.sd_bridge.transfer_parameters`.

M7 example: the ten nearest regions to EE00 on all 16 NUTS-2 indicators and on the lens
(GDP, household income, density, current cars per 1,000) share no region.
"""

from __future__ import annotations

import pandas as pd


def lens_for(registry, target: str, include_state: bool = True, moderators: list[str] | None = None) -> list[str]:
    """Indicators forming the lens: parents of `target`, listed moderators, and the target
    itself when `include_state` is true (saturation and level effects)."""
    parents = registry.parents(target)
    return list(dict.fromkeys(parents + (moderators or []) + ([target] if include_state else [])))


def s_causal(model, registry, geo_id: str, target: str, k: int = 10, **kwargs) -> pd.DataFrame:
    """k nearest entities on the causal lens of `target`, with distances and the overlap
    (Jaccard) with general similarity neighbours."""
    raise NotImplementedError("M7.3")
