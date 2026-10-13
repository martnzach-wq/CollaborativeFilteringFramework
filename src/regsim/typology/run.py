"""Typology runs (design §6)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypologyRun:
    lens: object
    levels: tuple[int, ...]
    representation: str = "whitened_profile"    # or "latent"
    algorithm: str = "gmm"                      # gmm | ward | hdbscan | ward_spatial | cocluster
    k: int | None = None                        # None = choose by stability and BIC
    k_range: tuple[int, int] = (3, 12)
    n_boot: int = 50
    seed: int = 0


def run_typology(model, run: TypologyRun):
    """Returns labels, soft memberships, stability (ARI, consensus matrix), centroid profiles, auto labels."""
    raise NotImplementedError("M5")


def track(typology_a, typology_b):
    """Hungarian alignment of clusters across periods/levels/runs; transition matrix."""
    raise NotImplementedError("M5")
