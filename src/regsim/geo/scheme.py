"""Geo-scheme abstraction (design §2.2). Models see only a weighted directed acyclic graph of units."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Protocol

import numpy as np
from scipy import sparse

from regsim.schema import GeoEdge, GeoUnit


class GeoScheme(Protocol):
    name: str

    def units(self, level: int | None = None) -> list[GeoUnit]: ...
    def edges(self) -> list[GeoEdge]: ...
    def harmonise(self, codes: Iterable[str], source_version: str) -> dict[str, list[tuple[str, float]]]: ...


def qualify(scheme: str, code: str) -> str:
    """Scheme-qualified identifier, e.g. ``qualify("NUTS2024", "AT130") == "NUTS2024:AT130"``."""
    return f"{scheme}:{code}"


def split_geo_id(geo_id: str) -> tuple[str, str]:
    scheme, _, code = geo_id.partition(":")
    return scheme, code


class HierarchyGraph:
    """Union of units and parent-child edges across schemes, with population and area weights.

    Edge weights are shares of the child in the parent (``weight_pop``, ``weight_area``). A child may have
    several parents with fractional membership; the membership of child c in parent p is
    ``weight_pop * population(p) / population(c)`` and equals 1 for nested schemes.
    """

    def __init__(self, units: Iterable[GeoUnit], edges: Iterable[GeoEdge]):
        self.units = {u.geo_id: u for u in units}
        self.edges = [e for e in edges if e.child in self.units and e.parent in self.units]
        self._parents: dict[str, list[GeoEdge]] = defaultdict(list)
        self._children: dict[str, list[GeoEdge]] = defaultdict(list)
        for e in self.edges:
            self._parents[e.child].append(e)
            self._children[e.parent].append(e)

    def __contains__(self, geo_id: str) -> bool:
        return geo_id in self.units

    def __len__(self) -> int:
        return len(self.units)

    def level(self, geo_id: str) -> int:
        return self.units[geo_id].level

    def parents(self, geo_id: str, weight: str = "pop") -> list[tuple[str, float]]:
        return [(e.parent, self._weight(e, weight)) for e in self._parents.get(geo_id, [])]

    def children(self, geo_id: str, weight: str = "pop") -> list[tuple[str, float]]:
        return [(e.child, self._weight(e, weight)) for e in self._children.get(geo_id, [])]

    def ancestors(self, geo_id: str) -> list[str]:
        """All ancestors, nearest first (breadth-first over parent edges)."""
        seen: dict[str, None] = {}
        queue = deque(p for p, _ in self.parents(geo_id))
        while queue:
            p = queue.popleft()
            if p in seen:
                continue
            seen[p] = None
            queue.extend(q for q, _ in self.parents(p))
        return list(seen)

    def descendants(self, geo_id: str) -> list[str]:
        seen: dict[str, None] = {}
        queue = deque(c for c, _ in self.children(geo_id))
        while queue:
            c = queue.popleft()
            if c in seen:
                continue
            seen[c] = None
            queue.extend(d for d, _ in self.children(c))
        return list(seen)

    def membership(self, child: str, parent: str) -> float:
        """Fraction of the child that lies in the parent."""
        for e in self._parents.get(child, []):
            if e.parent == parent:
                return self._weight(e, "membership")
        return 0.0

    def aggregation_matrix(
        self, index: list[str], weight: str = "pop"
    ) -> tuple[sparse.csr_matrix, list[int]]:
        """Aggregation of children to parents within ``index``.

        Parameters
        ----------
        index : list of str
            Row order of the entity matrix.
        weight : {"pop", "area", "membership"}
            ``pop`` and ``area`` give weighted means (rows renormalised over the children present in
            ``index``); ``membership`` gives sums of extensive quantities (not renormalised).

        Returns
        -------
        W : scipy.sparse.csr_matrix, shape (len(index), len(index))
            ``W[p, c]`` is the weight of child c in parent p; rows of units without children are empty.
        parent_rows : list of int
            Rows of ``index`` that have at least one child in ``index``.
        """
        pos = {g: i for i, g in enumerate(index)}
        rows, cols, vals = [], [], []
        for p in index:
            kids = [(pos[c], w) for c, w in self.children(p, weight) if c in pos]
            if not kids:
                continue
            total = sum(w for _, w in kids) if weight in ("pop", "area") else 1.0
            for c, w in kids:
                rows.append(pos[p])
                cols.append(c)
                vals.append(w / total if total > 0 else 1.0 / len(kids))
        n = len(index)
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return W, sorted(set(rows))

    def parent_index(self, index: list[str]) -> list[list[tuple[int, float]]]:
        """Parents of each row restricted to ``index``, as ``(row, population share)`` (``ModelData.parent``)."""
        pos = {g: i for i, g in enumerate(index)}
        return [[(pos[p], w) for p, w in self.parents(g) if p in pos] for g in index]

    def subgraph(self, geo_ids: Iterable[str]) -> HierarchyGraph:
        keep = set(geo_ids)
        return HierarchyGraph([u for g, u in self.units.items() if g in keep], self.edges)

    def _weight(self, e: GeoEdge, weight: str) -> float:
        if weight == "pop":
            return e.weight_pop
        if weight == "area":
            return e.weight_area
        if weight == "membership":
            pc = self.units[e.child].population
            pp = self.units[e.parent].population
            if pc and pp and np.isfinite(pc) and np.isfinite(pp):
                return float(min(1.0, e.weight_pop * pp / pc))
            ac, ap = self.units[e.child].area_km2, self.units[e.parent].area_km2
            if ac and ap:
                return float(min(1.0, e.weight_area * ap / ac))
            return 1.0
        raise ValueError(f"unknown weight '{weight}'")
