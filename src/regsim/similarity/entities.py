"""Geo-entity similarity (design §5.1): S-profile, S-latent, S-gower, percentile scores, contributions."""

from __future__ import annotations


def s_profile(model, lens, ids=None):
    """Whitened distance of completed profiles on the lens; returns distances, intervals, contributions."""
    raise NotImplementedError("M4")


def s_gower(observations, lens, d36_mode: bool = False):
    """Gower distance on observed values with coverage; d36_mode reproduces KNOWING D3.6 scoring."""
    raise NotImplementedError("M4")


def nearest(model, geo_id, lens, k: int = 10, measure: str = "profile"):
    raise NotImplementedError("M4")
