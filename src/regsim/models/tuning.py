"""Hyperparameters by masked cross-validation within the training cells (nested inside the benchmark)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from regsim.evaluation import masking
from regsim.models.base import ModelData


class Tuned:
    """Choose hyperparameters of ``estimator`` by masked cross-validation, then refit on all training cells.

    Parameters
    ----------
    estimator : type
        Estimator class.
    grid : dict of str to list
        Candidate values per hyperparameter; all combinations are tried.
    base : dict, optional
        Fixed hyperparameters.
    n_folds : int
        Number of inner random-cell splits.
    frac : float
        Share of training cells held out per inner split.
    seed : int
        Seed of the inner splits.
    """

    def __init__(
        self,
        estimator,
        grid: dict[str, list],
        base: dict | None = None,
        n_folds: int = 2,
        frac: float = 0.1,
        seed: int = 0,
    ):
        self.estimator = estimator
        self.grid = grid
        self.base = base or {}
        self.n_folds = n_folds
        self.frac = frac
        self.seed = seed

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    def fit(self, data: ModelData) -> Tuned:
        keys = list(self.grid)
        combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*self.grid.values())]
        splits = [masking.random_cells(data.mask, self.frac, self.seed + f) for f in range(self.n_folds)]
        rows = []
        for params in combos:
            errs = []
            for s in splits:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = self.estimator(**self.base, **params).fit(data.with_mask(s.train(data.mask)))
                e = (m.predict() - data.Z)[s.test]
                errs.append(float(np.sqrt(np.mean(e**2))))
            rows.append({**params, "rmse": float(np.mean(errs))})
        self.cv_results_ = pd.DataFrame(rows)
        best = int(self.cv_results_.rmse.idxmin())
        self.best_params_ = combos[best]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_estimator_ = self.estimator(**self.base, **self.best_params_).fit(data)
        return self

    def __getattr__(self, name):
        if name.endswith("_") or name.startswith("__") or "best_estimator_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.best_estimator_, name)

    def predict(self) -> np.ndarray:
        return self.best_estimator_.predict()

    def predict_dist(self, n_samples: int = 200, seed: int | np.random.Generator = 0) -> np.ndarray:
        return self.best_estimator_.predict_dist(n_samples, seed)

    def fold_in(self, *args, **kwargs):
        return self.best_estimator_.fold_in(*args, **kwargs)

    def embed_entities(self) -> np.ndarray:
        return self.best_estimator_.embed_entities()

    def embed_indicators(self) -> np.ndarray:
        return self.best_estimator_.embed_indicators()
