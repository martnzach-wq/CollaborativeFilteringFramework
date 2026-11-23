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
"""

from __future__ import annotations

import pandas as pd


def load_events(path: str) -> pd.DataFrame:
    raise NotImplementedError("M7.5")


def effect(model, events: pd.DataFrame, outcome: str, analogues: int = 20, placebo: int = 200, seed: int = 0) -> pd.DataFrame:
    """Estimated effect per treated entity and post year, with placebo p-values."""
    raise NotImplementedError("M7.5")
