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
  (type `effect`, horizon short run). Not used as inputs to HierTypedMF.
- `s_effect`: effect distance and probability of practical equivalence
  P(|beta_A - beta_B| < delta) between two entities, using the joint covariance of
  the predictions.

Moderator sets and region types come from configuration and `regsim.typology`, not
from geo-code prefixes.
"""

from __future__ import annotations

import pandas as pd


def fit_varying(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    moderators: pd.DataFrame,
    alpha: float | None = None,
    **kwargs,
):
    """Fit the varying-coefficient model; `alpha=None` selects the penalty by leave-one-group-out."""
    raise NotImplementedError("M7.6")


def validate_gate(
    panel: pd.DataFrame,
    cause: str,
    effect: str,
    moderators: pd.DataFrame,
    groups: pd.Series,
    threshold: float = 0.9,
) -> dict:
    """Leave-one-group-out comparison with the pooled effect; returns relative error and decision."""
    raise NotImplementedError("M7.6")


def region_effects(fit, entities: pd.Index, level: float = 0.9) -> pd.DataFrame:
    raise NotImplementedError("M7.6")


def s_effect(fits: dict, geo_a: str, geo_b: str, delta: float = 0.1) -> pd.DataFrame:
    """Per-link difference, standard error and probability of practical equivalence."""
    raise NotImplementedError("M7.6")
