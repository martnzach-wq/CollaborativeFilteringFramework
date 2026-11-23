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


def lens_for(
    registry, target: str, include_state: bool = True, moderators: list[str] | None = None
) -> list[str]:
    """Indicators forming the lens: parents of `target`, listed moderators, and the target
    itself when `include_state` is true (saturation and level effects)."""
    parents = registry.parents(target)
    return list(dict.fromkeys(parents + (moderators or []) + ([target] if include_state else [])))


def s_causal(
    model,
    registry,
    geo_id: str,
    target: str,
    k: int = 10,
    include_state: bool = True,
    moderators: list[str] | None = None,
    data=None,
) -> pd.DataFrame:
    """k nearest entities on the causal lens of `target`, with distances and the overlap
    (Jaccard) with general similarity neighbours.

    Parameters
    ----------
    model : Fitted or Estimator
    registry : EdgeRegistry
    geo_id, target : str
    data : ModelData, optional
        The model's data (default: taken from ``model``).

    Returns
    -------
    DataFrame
        Output of :func:`regsim.similarity.entities.nearest` on the lens; ``attrs`` hold the ``lens``,
        the general neighbours and their ``jaccard`` overlap.
    """
    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, s_profile

    fitted = model
    model = getattr(fitted, "model", fitted)
    data = data if data is not None else getattr(fitted, "data", None)
    if data is None:
        data = getattr(model, "estimator_", model).data_
    level = int(data.levels()[data.geo_ids.index(geo_id)])
    lens = [c for c in lens_for(registry, target, include_state, moderators) if c in data.column_ids]
    if not lens:
        raise ValueError(f"no indicator of the causal lens of {target!r} is in the model")
    causal = nearest(s_profile(model, data, Lens(indicators=tuple(lens), levels=(level,))), geo_id, k=k)
    general = nearest(s_profile(model, data, Lens(levels=(level,))), geo_id, k=k)
    a, b = set(causal.geo_id), set(general.geo_id)
    causal.attrs.update(
        lens=lens,
        target=target,
        general_neighbours=list(general.geo_id),
        jaccard=len(a & b) / len(a | b) if a | b else float("nan"),
    )
    return causal
