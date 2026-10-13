"""Masking schemes M1–M5 (design §10). Each function returns test masks; training is ``mask & ~test``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    """One train/test split of the observed cells."""

    scheme: str
    test: np.ndarray
    seed: int
    k: int | None = None
    column: str | None = None
    group: str | None = None

    def train(self, mask: np.ndarray) -> np.ndarray:
        return mask & ~self.test


def random_cells(mask: np.ndarray, frac: float = 0.1, seed: int = 0) -> Split:
    """M1: a random share ``frac`` of the observed cells."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    pick = idx[rng.choice(len(idx), int(frac * len(idx)), replace=False)]
    test = np.zeros_like(mask)
    test[pick[:, 0], pick[:, 1]] = True
    return Split("M1 random cells", test, seed)


def entity_cold_start(
    mask: np.ndarray,
    level: np.ndarray,
    target_level: int,
    frac: float = 0.15,
    k_keep: tuple[int, ...] = (1, 3, 6, 10),
    min_observed: int = 15,
    seed: int = 0,
) -> list[Split]:
    """M2: for a share ``frac`` of the entities at ``target_level`` with at least ``min_observed`` cells, hide
    all but ``k`` observed cells, for each ``k`` in ``k_keep``.

    The same entities are used for all ``k``; the kept cells are drawn independently per ``k``, in the order of
    ``k_keep`` from one generator (the prototype's protocol).
    """
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero((level == target_level) & (mask.sum(axis=1) >= min_observed))
    pick = rng.choice(cand, int(frac * len(cand)), replace=False)
    out = []
    for k in k_keep:
        test = np.zeros_like(mask)
        for g in pick:
            obs = np.flatnonzero(mask[g])
            keep = rng.choice(obs, k, replace=False)
            test[g, np.setdiff1d(obs, keep)] = True
        out.append(Split(f"M2 cold start (level {target_level})", test, seed, k=k))
    return out


def country_blocks(
    mask: np.ndarray,
    level: np.ndarray,
    country: np.ndarray,
    columns: list[int],
    target_level: int,
    n_blocks: int = 10,
    min_cells: int = 4,
    seed: int = 0,
) -> Split:
    """M3: hide ``n_blocks`` (country, column) blocks at ``target_level`` (missing not at random).

    Candidate blocks have at least ``min_cells`` observed cells; ``columns`` restricts the columns (e.g. those
    native at ``target_level``).
    """
    rng = np.random.default_rng(seed)
    at = level == target_level
    cands = [
        (c, j)
        for c in np.unique(country)
        for j in columns
        if (mask[at & (country == c), j]).sum() >= min_cells
    ]
    test = np.zeros_like(mask)
    for i in rng.choice(len(cands), min(n_blocks, len(cands)), replace=False):
        c, j = cands[i]
        test[at & (country == c) & mask[:, j], j] = True
    return Split("M3 country blocks", test, seed)


def level_transfer(
    mask: np.ndarray, level: np.ndarray, column: int, target_level: int, name: str | None = None
) -> Split:
    """M4: hide all values of one column at ``target_level``; coarser values stay observed."""
    test = np.zeros_like(mask)
    rows = level == target_level
    test[rows, column] = mask[rows, column]
    return Split(f"M4 level transfer (to level {target_level})", test, 0, column=name or str(column))


def temporal(mask, periods, last=1):
    raise NotImplementedError("M6")
