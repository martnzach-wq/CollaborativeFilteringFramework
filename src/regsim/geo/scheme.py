"""Geo-scheme abstraction (design §2.2). Models see only a weighted directed acyclic graph of units."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from regsim.schema import GeoEdge, GeoUnit


class GeoScheme(Protocol):
    name: str

    def units(self, level: int | None = None) -> list[GeoUnit]: ...
    def edges(self) -> list[GeoEdge]: ...
    def harmonise(self, codes: Iterable[str], source_version: str) -> dict[str, list[tuple[str, float]]]: ...


class HierarchyGraph:
    """Union of units and parent-child edges across schemes, with population and area weights."""

    def __init__(self, units: list[GeoUnit], edges: list[GeoEdge]):
        self.units = {u.geo_id: u for u in units}
        self.edges = edges

    def parents(self, geo_id: str) -> list[tuple[str, float]]:
        raise NotImplementedError("M1")

    def children(self, geo_id: str) -> list[tuple[str, float]]:
        raise NotImplementedError("M1")

    def ancestors(self, geo_id: str) -> list[str]:
        raise NotImplementedError("M1")

    def aggregation_matrix(self, index: list[str], weight: str = "pop"):
        """Sparse matrix W (parents x index) with child weights normalised per parent."""
        raise NotImplementedError("M1")
