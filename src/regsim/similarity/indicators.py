"""Indicator similarity (design §5.2): I-model, I-observed, I-partial, I-predictive."""

from __future__ import annotations


def i_model(model, subset=None):
    raise NotImplementedError("M4")


def i_observed(observations, level, min_pairs: int = 30):
    raise NotImplementedError("M4")


def i_partial(model, alpha: float = 0.05):
    raise NotImplementedError("M4")


def i_predictive(model, target, candidates):
    raise NotImplementedError("M4")
