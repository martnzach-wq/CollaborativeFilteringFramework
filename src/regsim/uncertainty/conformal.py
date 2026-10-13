"""Split-conformal recalibration of predictive intervals with Mondrian groups (design §8).

Calibration cells are held out from the training cells in two ways that mirror the evaluation schemes: random
cells (as M1) and cold-start entities that keep only a few cells (as M2). The model is fitted on the remaining
cells, and the normalised scores |z - zhat| / sigma on the calibration cells give one conformal quantile per
Mondrian group (level x number of observed indicators of the entity). The calibrated standard deviation is
sigma * q_group / z_{1 - alpha / 2}, so that the Gaussian interval with the calibrated scale is the conformal
interval.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from regsim.evaluation import masking
from regsim.models.base import ModelData

DEFAULT_BINS = (3, 7, 12)  # observed indicators per entity: <=3, 4-7, 8-12, >=13


def model_sigma(model, shape: tuple[int, int]) -> np.ndarray:
    """Predictive standard deviation of a fitted model: ``predict_std`` or the residual scale per column."""
    if hasattr(model, "predict_std"):
        s = np.asarray(model.predict_std(), dtype=float)
    elif getattr(model, "scale_", None) is not None:
        s = np.broadcast_to(np.asarray(model.scale_, dtype=float), shape).copy()
    else:
        s = np.ones(shape)
    return s


def mondrian_groups(level: np.ndarray, mask: np.ndarray, bins=DEFAULT_BINS) -> np.ndarray:
    """Group id per entity: (level, bin of the number of observed indicators)."""
    nobs = mask.sum(axis=1)
    return np.asarray(level) * 10 + np.digitize(nobs, np.asarray(bins) + 0.5)


@dataclass
class ConformalCalibrator:
    """Conformal quantiles of normalised scores per group, with fallbacks."""

    alpha: float
    quantiles: dict[int, float] = field(default_factory=dict)
    by_bin: dict[int, float] = field(default_factory=dict)
    overall: float = float("nan")
    counts: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def _q(scores: np.ndarray, alpha: float) -> float:
        n = scores.size
        k = int(np.ceil((n + 1) * (1 - alpha)))
        return float(np.sort(scores)[min(k, n) - 1]) if n else float("nan")

    @classmethod
    def fit(cls, scores: np.ndarray, groups: np.ndarray, alpha: float = 0.1, min_count: int = 50):
        """Quantiles per group from ``scores`` (one per calibration cell) and their entity ``groups``."""
        ok = np.isfinite(scores)
        scores, groups = scores[ok], groups[ok]
        cal = cls(alpha, overall=cls._q(scores, alpha))
        for g in np.unique(groups):
            s = scores[groups == g]
            cal.counts[int(g)] = int(s.size)
            if s.size >= min_count:
                cal.quantiles[int(g)] = cls._q(s, alpha)
        for b in np.unique(groups % 10):
            s = scores[groups % 10 == b]
            if s.size >= min_count:
                cal.by_bin[int(b)] = cls._q(s, alpha)
        return cal

    def quantile(self, groups: np.ndarray) -> np.ndarray:
        """Quantile per entity group (group, then observed-count bin, then overall)."""
        return np.array(
            [self.quantiles.get(int(g), self.by_bin.get(int(g) % 10, self.overall)) for g in np.ravel(groups)]
        ).reshape(np.shape(groups))


def calibration_split(
    mask: np.ndarray,
    level: np.ndarray,
    cell_frac: float = 0.1,
    entity_frac: float = 0.15,
    k_keep: tuple[int, ...] = (1, 3, 6, 10),
    min_observed: int = 15,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Calibration cells: ``cell_frac`` random cells plus cold-start entities (a share ``entity_frac`` of the
    entities with at least ``min_observed`` cells at each level keep ``k`` cells, ``k`` drawn from ``k_keep``)."""
    rng = np.random.default_rng(seed)
    cal = np.zeros_like(mask)
    for lv in np.unique(level):
        cand = np.flatnonzero((level == lv) & (mask.sum(axis=1) >= min_observed))
        for g in rng.choice(cand, int(round(entity_frac * cand.size)), replace=False):
            obs = np.flatnonzero(mask[g])
            keep = rng.choice(obs, int(rng.choice(k_keep)), replace=False)
            cal[g, np.setdiff1d(obs, keep)] = True
    cells = masking.random_cells(mask & ~cal, cell_frac, int(rng.integers(2**31))).test
    return cal | cells


class Conformalised:
    """Estimator wrapper with split-conformal calibrated intervals (Mondrian by level and observed count).

    ``n_splits`` models are fitted, each without its own calibration cells; their normalised scores are pooled
    for the quantiles, and predictions and standard deviations are the means over the models. Pooling reduces
    the variance of the quantiles; a model refitted on all cells would be more accurate than the calibration
    models and its intervals too wide (``refit=True`` keeps that option).

    Parameters
    ----------
    factory : callable
        Returns an unfitted estimator.
    alpha : float
        Miscoverage of the reference interval (0.1 for 90 % intervals).
    n_splits : int
        Number of calibration splits and models.
    refit : bool
        Predict with one model refitted on all training cells instead of the calibration models.
    bins : tuple of int
        Upper edges of the observed-count bins.
    cell_frac, entity_frac : float
        Shares of random calibration cells and of cold-start calibration entities per split.
    min_count : int
        Minimum number of calibration cells for a group quantile.
    seed : int
        Seed of the calibration splits.
    """

    def __init__(
        self,
        factory: Callable[[], object],
        alpha: float = 0.1,
        n_splits: int = 3,
        refit: bool = False,
        bins: tuple[int, ...] = DEFAULT_BINS,
        cell_frac: float = 0.1,
        entity_frac: float = 0.15,
        min_count: int = 50,
        seed: int = 0,
    ):
        self.factory = factory
        self.alpha = alpha
        self.n_splits = n_splits
        self.refit = refit
        self.bins = bins
        self.cell_frac = cell_frac
        self.entity_frac = entity_frac
        self.min_count = min_count
        self.seed = seed

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    def _fit_one(self, data: ModelData, mask: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.factory().fit(data.with_mask(mask))

    def fit(self, data: ModelData) -> Conformalised:
        level = data.levels()
        rng = np.random.default_rng(self.seed)
        scores, groups, models, cals = [], [], [], []
        for _ in range(self.n_splits):
            cal = calibration_split(data.mask, level, self.cell_frac, self.entity_frac, seed=rng)
            fit_mask = data.mask & ~cal
            model = self._fit_one(data, fit_mask)
            mu, sd = model.predict(), model_sigma(model, data.Z.shape)
            scores.append((np.abs(data.Z - mu) / sd)[cal])
            groups.append(
                np.broadcast_to(mondrian_groups(level, fit_mask, self.bins)[:, None], data.Z.shape)[cal]
            )
            models.append(model)
            cals.append(cal)
        self.calibrator_ = ConformalCalibrator.fit(
            np.concatenate(scores), np.concatenate(groups), self.alpha, self.min_count
        )
        self.models_ = [self._fit_one(data, data.mask)] if self.refit else models
        self.estimator_ = self.models_[0]
        self.groups_ = mondrian_groups(level, data.mask, self.bins)
        self.calibration_masks_ = cals
        return self

    def __getattr__(self, name):
        if name.endswith("_") or name.startswith("__") or "estimator_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.estimator_, name)

    def predict(self) -> np.ndarray:
        return np.mean([m.predict() for m in self.models_], axis=0)

    def _sigma(self) -> np.ndarray:
        shape = self.estimator_.predict().shape
        return np.mean([model_sigma(m, shape) for m in self.models_], axis=0)

    def predict_std(self) -> np.ndarray:
        """Calibrated standard deviation: sigma * q_group / z_{1 - alpha / 2}."""
        q = self.calibrator_.quantile(self.groups_)[:, None]
        return self._sigma() * q / norm.ppf(1 - self.alpha / 2)

    def predict_interval(self, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated interval; other levels than the calibrated ``alpha`` assume Gaussian score shape."""
        z = norm.ppf(1 - (alpha or self.alpha) / 2)
        mu, sd = self.predict(), self.predict_std()
        return mu - z * sd, mu + z * sd

    def predict_dist(self, n_samples: int = 200, seed: int | np.random.Generator = 0) -> np.ndarray:
        """Samples of the first model, rescaled around the point prediction by the calibration factor."""
        S = self.estimator_.predict_dist(n_samples, seed)
        mu0, mu = self.estimator_.predict(), self.predict()
        factor = self.predict_std() / model_sigma(self.estimator_, mu.shape)
        return mu + (S - mu0) * np.where(np.isfinite(factor), factor, 1.0)

    def fold_in(self, *args, **kwargs):
        return self.estimator_.fold_in(*args, **kwargs)

    def embed_entities(self) -> np.ndarray:
        return self.estimator_.embed_entities()

    def embed_indicators(self) -> np.ndarray:
        return self.estimator_.embed_indicators()
