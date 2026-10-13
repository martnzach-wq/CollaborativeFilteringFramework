"""Adapter for the KNOWING D3.6 GeoPackage (doi:10.5281/zenodo.20376109)."""
from __future__ import annotations


def read_d36(gpkg_path: str):
    """Read `knowing.indicator`, `indicator_values`, `regions` and `precomputed_scores`; return
    (observations, geo units, indicator registry entries, published scores)."""
    raise NotImplementedError("M1")
