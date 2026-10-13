"""Estimator interface shared by all models (design §4.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ModelData:
    Z: np.ndarray  # entities x columns, transformed + scaled, NaN = missing
    mask: np.ndarray  # observed cells used for fitting
    parent: list[list[tuple[int, float]]]  # fractional parents per row
    agg: object  # sparse aggregation matrix (parents x rows) on original scale
    column_meta: list  # Indicator per column (after CLR expansion)
    side: np.ndarray | None = None  # side information x_g


class Estimator(Protocol):
    def fit(self, data: ModelData) -> Estimator: ...
    def predict(self) -> np.ndarray: ...
    def predict_dist(self, n_samples: int = 200) -> np.ndarray: ...
    def fold_in(
        self, z_new: np.ndarray, mask_new: np.ndarray, parent_new=None, side_new=None
    ) -> np.ndarray: ...
    def embed_entities(self) -> np.ndarray: ...
    def embed_indicators(self) -> np.ndarray: ...
