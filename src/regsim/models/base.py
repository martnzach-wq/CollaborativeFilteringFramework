"""Estimator interface shared by all models (design §4.5)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np


@dataclass
class ModelData:
    """Model-ready matrix on the transformed and robust-scaled scale.

    Attributes
    ----------
    Z : ndarray, shape (n_entities, n_columns)
        Values on the model scale; NaN where missing.
    mask : ndarray of bool
        Cells used for fitting (observed and not held out).
    parent : list of list of (int, float)
        Parents of each row within the matrix, with the fraction of the child lying in each parent (sums to one
        per child).
    agg : scipy.sparse matrix, shape (n_entities, n_entities)
        Population-weighted aggregation of children to parents (row p holds the shares of p's children).
    agg_area, agg_sum : scipy.sparse matrix, optional
        Area-weighted means and membership-weighted sums of children.
    agg_cover : ndarray, optional
        Share of each parent's population covered by its children in the matrix.
    column_meta : list of Indicator
        Indicator per column (after CLR expansion).
    side : ndarray, optional
        Side information x_g.
    level : ndarray of int
        Level of each row.
    country : ndarray of str
        Top-level ancestor of each row, used for grouped masking.
    geo_ids, column_ids : list of str
        Row and column labels.
    scalers : list of (transform, location, scale)
        Inverse mapping to the original scale per column.
    side_names : list of str
        Names of the side-information columns.
    adjacency : scipy.sparse matrix, optional
        Symmetric spatial adjacency of rows at the same level (for the Laplacian penalty).
    period : ndarray of str, optional
        Period window of each row (geo-time entities).
    time_pairs : ndarray of int, shape (n_pairs, 2), optional
        Rows of the same entity in consecutive windows (random-walk penalty).
    """

    Z: np.ndarray
    mask: np.ndarray
    parent: list[list[tuple[int, float]]]
    agg: object
    column_meta: list
    side: np.ndarray | None = None
    level: np.ndarray | None = None
    country: np.ndarray | None = None
    geo_ids: list[str] = field(default_factory=list)
    column_ids: list[str] = field(default_factory=list)
    scalers: list[tuple[str, float, float]] = field(default_factory=list)
    agg_area: object = None
    agg_sum: object = None
    agg_cover: np.ndarray | None = None
    side_names: list[str] = field(default_factory=list)
    adjacency: object = None
    period: np.ndarray | None = None
    time_pairs: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.Z.shape

    def with_mask(self, mask: np.ndarray) -> ModelData:
        """Copy that fits on ``mask`` (cells outside it are hidden from the estimator)."""
        return replace(self, Z=np.where(mask, self.Z, np.nan), mask=mask)

    def levels(self) -> np.ndarray:
        return self.level if self.level is not None else np.zeros(self.Z.shape[0], dtype=int)


class Estimator(Protocol):
    def fit(self, data: ModelData) -> Estimator: ...
    def predict(self) -> np.ndarray: ...
    def predict_dist(self, n_samples: int = 200) -> np.ndarray: ...
    def fold_in(
        self, z_new: np.ndarray, mask_new: np.ndarray, parent_new=None, side_new=None
    ) -> np.ndarray: ...
    def embed_entities(self) -> np.ndarray: ...
    def embed_indicators(self) -> np.ndarray: ...
