"""Neighbour-prediction skill (design §10): a similarity is better if its nearest neighbours predict held-out
indicators better."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regsim.evaluation.masking import Split
from regsim.models.base import ModelData
from regsim.similarity.entities import ProfileSpace, noise_variance, s_gower


def neighbour_predictions(data: ModelData, split: Split, distance_rows, k: int = 10) -> np.ndarray:
    """Predict each test cell as the inverse-distance weighted mean of the training values of the ``k`` nearest
    entities at the same level that observe the column; ``distance_rows(g, rows)`` gives distances."""
    train = split.train(data.mask)
    level = data.levels()
    P = np.full(data.Z.shape, np.nan)
    for g in np.flatnonzero(split.test.any(axis=1)):
        rows = np.flatnonzero(level == level[g])
        rows = rows[rows != g]
        d = distance_rows(g, rows)
        for j in np.flatnonzero(split.test[g]):
            ok = train[rows, j] & np.isfinite(d)
            if not ok.any():
                continue
            cand, dc = rows[ok], d[ok]
            nn = np.argsort(dc)[:k]
            w = 1.0 / (dc[nn] + 1e-3)
            P[g, j] = np.sum(w * data.Z[cand[nn], j]) / w.sum()
    return P


def neighbour_skill(data: ModelData, splits: list[Split], model_factory, k: int = 10) -> pd.DataFrame:
    """RMSE of neighbour predictions under S-profile and S-gower per split (lower = more skilful)."""
    rows = []
    for s in splits:
        train = s.train(data.mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_factory().fit(data.with_mask(train))
        space = ProfileSpace(model.predict(), data, noise_var=noise_variance(model, data))
        measures = {
            "S-profile": lambda g, r, sp=space: sp.distances(g, r),
            "S-gower": lambda g, r, tr=train: s_gower(data, g, r, mask=tr)[0],
        }
        for name, fn in measures.items():
            P = neighbour_predictions(data, s, fn, k)
            ok = s.test & np.isfinite(P)
            rows.append(
                {
                    "scheme": s.scheme,
                    "seed": s.seed,
                    "k_known": s.k,
                    "measure": name,
                    "n_cells": int(ok.sum()),
                    "coverage": float(ok.sum() / s.test.sum()),
                    "rmse": float(np.sqrt(np.mean((P - data.Z)[ok] ** 2))),
                }
            )
    return pd.DataFrame(rows)
