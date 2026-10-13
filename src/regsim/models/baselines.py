"""Baselines (design §4.5): GlobalMean, LevelMean, ParentCarryDown, KNNGower, IterativePCA, SoftImpute, MissForest.

All baselines predict every cell of the matrix on the model scale. Their predictive distribution is Gaussian
around the point prediction with a per-column scale equal to the in-sample root mean squared residual (floored
at 10 % of the column's standard deviation); fold-in appends the new rows and refits.
"""

from __future__ import annotations

import numpy as np

from regsim.models.base import ModelData


class _Baseline:
    """Shared fit/predict machinery; subclasses implement ``_complete``."""

    def fit(self, data: ModelData) -> _Baseline:
        self.data_ = data
        self.P_ = self._complete(data)
        if self.P_.shape != data.Z.shape or not np.all(np.isfinite(self.P_)):
            raise RuntimeError(f"{type(self).__name__} produced a non-finite or misshaped prediction")
        resid = np.where(data.mask, self.P_ - data.Z, np.nan)
        with np.errstate(all="ignore"):
            rms = np.sqrt(np.nanmean(resid**2, axis=0))
            sd = np.nanstd(np.where(data.mask, data.Z, np.nan), axis=0)
        floor = 0.1 * np.where(np.isfinite(sd), sd, 1.0)
        self.scale_ = np.maximum(np.where(np.isfinite(rms), rms, 1.0), floor)
        return self

    def predict(self) -> np.ndarray:
        return self.P_

    def predict_dist(self, n_samples: int = 200, seed: int | np.random.Generator = 0) -> np.ndarray:
        """Samples of shape ``(n_samples, n_entities, n_columns)``."""
        rng = np.random.default_rng(seed)
        return self.P_[None] + self.scale_[None, None] * rng.standard_normal((n_samples, *self.P_.shape))

    def fold_in(self, z_new, mask_new, parent_new=None, side_new=None, level_new=None, country_new=None):
        """Predictions for new rows, obtained by refitting on the extended matrix."""
        d = self.data_
        z_new = np.atleast_2d(np.asarray(z_new, dtype=float))
        mask_new = np.atleast_2d(np.asarray(mask_new, dtype=bool))
        m = z_new.shape[0]
        parent_new = parent_new or [[] for _ in range(m)]
        if level_new is None:
            lv = d.levels()
            level_new = [lv[p[0][0]] + 1 if p else 0 for p in parent_new]
        country = None
        if d.country is not None:
            country = np.concatenate([d.country, country_new if country_new is not None else ["?"] * m])
        ext = ModelData(
            Z=np.vstack([np.where(d.mask, d.Z, np.nan), np.where(mask_new, z_new, np.nan)]),
            mask=np.vstack([d.mask, mask_new]),
            parent=list(d.parent) + [list(p) for p in parent_new],
            agg=None,
            column_meta=d.column_meta,
            level=np.concatenate([d.levels(), np.asarray(level_new, dtype=int)]),
            country=country,
            geo_ids=list(d.geo_ids) + [f"new:{i}" for i in range(m)],
            column_ids=d.column_ids,
            scalers=d.scalers,
        )
        return type(self)(**self.get_params()).fit(ext).predict()[-m:]

    def embed_entities(self) -> np.ndarray:
        return self.P_

    def embed_indicators(self) -> np.ndarray:
        return self.P_.T

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    def _complete(self, data: ModelData) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


def _train(data: ModelData) -> np.ndarray:
    return np.where(data.mask, data.Z, np.nan)


def _col_means(Zt: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        mu = np.nanmean(Zt, axis=0)
    return np.where(np.isfinite(mu), mu, 0.0)


def _level_means(Zt: np.ndarray, level: np.ndarray) -> np.ndarray:
    glob = _col_means(Zt)
    P = np.empty_like(Zt)
    for lv in np.unique(level):
        rows = level == lv
        with np.errstate(all="ignore"):
            m = np.nanmean(Zt[rows], axis=0)
        P[rows] = np.where(np.isfinite(m), m, glob)
    return P


class GlobalMean(_Baseline):
    """Column mean of the training cells."""

    def _complete(self, data: ModelData) -> np.ndarray:
        return np.broadcast_to(_col_means(_train(data)), data.Z.shape).copy()


class LevelMean(_Baseline):
    """Column mean of the training cells at the same level; global mean where a level has none."""

    def _complete(self, data: ModelData) -> np.ndarray:
        return _level_means(_train(data), data.levels())


class ParentCarryDown(_Baseline):
    """Value of the nearest ancestor with a training value (LEVITATE copying); level mean otherwise.

    Rows with several parents take the population-share-weighted mean over the parents that carry a value.
    """

    def _complete(self, data: ModelData) -> np.ndarray:
        Zt = _train(data)
        level = data.levels()
        carried = np.full(Zt.shape, np.nan)  # value passed down from the nearest ancestor with data
        for g in np.argsort(level, kind="stable"):
            num = np.zeros(Zt.shape[1])
            den = np.zeros(Zt.shape[1])
            for p, w in data.parent[g]:
                v = np.where(np.isfinite(Zt[p]), Zt[p], carried[p])
                ok = np.isfinite(v)
                num[ok] += w * v[ok]
                den[ok] += w
            with np.errstate(all="ignore"):
                carried[g] = np.where(den > 0, num / den, np.nan)
        return np.where(np.isfinite(carried), carried, _level_means(Zt, level))


class KNNGower(_Baseline):
    """D3.6-type Gower nearest neighbours among rows of the same level.

    Distance: mean absolute difference of min–max scaled training values over the columns observed in both
    rows (pairwise deletion). A cell is the inverse-distance weighted mean of the ``k`` nearest rows of the same
    level that observe the column; the level mean where no such row exists.
    """

    def __init__(self, k: int = 10, eps: float = 1e-3):
        self.k = k
        self.eps = eps

    def _complete(self, data: ModelData) -> np.ndarray:
        Zt = _train(data)
        level = data.levels()
        with np.errstate(all="ignore"):
            span = np.nanmax(Zt, axis=0) - np.nanmin(Zt, axis=0)
        span = np.where(np.isfinite(span) & (span > 0), span, 1.0)
        X = Zt / span
        P = _level_means(Zt, level)
        for lv in np.unique(level):
            rows = np.flatnonzero(level == lv)
            D = self._distances(X[rows])
            XL = Zt[rows]
            for j in range(Zt.shape[1]):
                cand = np.flatnonzero(np.isfinite(XL[:, j]))
                if cand.size == 0:
                    continue
                Dj = D[:, cand]
                k = min(self.k, cand.size)
                nn = np.argpartition(Dj, k - 1, axis=1)[:, :k]
                d = np.take_along_axis(Dj, nn, axis=1)
                w = np.where(np.isfinite(d), 1.0 / (d + self.eps), 0.0)
                vals = XL[cand[nn], j]
                tot = w.sum(axis=1)
                with np.errstate(all="ignore"):
                    est = (w * vals).sum(axis=1) / tot
                P[rows, j] = np.where(tot > 0, est, P[rows, j])
        return P

    @staticmethod
    def _distances(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        total = np.zeros((n, n))
        count = np.zeros((n, n))
        for j in range(X.shape[1]):
            o = np.isfinite(X[:, j])
            if not o.any():
                continue
            x = np.where(o, X[:, j], 0.0)
            both = np.outer(o, o)
            total += np.where(both, np.abs(x[:, None] - x[None, :]), 0.0)
            count += both
        with np.errstate(all="ignore"):
            D = total / count
        D[count == 0] = np.inf
        np.fill_diagonal(D, np.inf)
        return D


class SoftImpute(_Baseline):
    """Nuclear-norm completion (Mazumder et al., 2010) on column-centred data; levels are independent rows."""

    def __init__(self, lam: float = 5.0, rank_max: int = 12, max_iter: int = 200, tol: float = 1e-4):
        self.lam = lam
        self.rank_max = rank_max
        self.max_iter = max_iter
        self.tol = tol

    def _complete(self, data: ModelData) -> np.ndarray:
        Zt = _train(data)
        mu = _col_means(Zt)
        X = np.where(data.mask, Zt - mu, 0.0)
        Zh = np.zeros_like(X)
        r = self.rank_max
        for _ in range(self.max_iter):
            U, s, Vt = np.linalg.svd(np.where(data.mask, X, Zh), full_matrices=False)
            s = np.maximum(s - self.lam, 0.0)[:r]
            new = (U[:, :r] * s) @ Vt[:r]
            done = np.linalg.norm(new - Zh) < self.tol * (np.linalg.norm(Zh) + 1e-9)
            Zh = new
            if done:
                break
        return Zh + mu


class IterativePCA(_Baseline):
    """Regularised iterative PCA (missMDA-type; Josse and Husson, 2012).

    Missing cells start at the column mean; each iteration fits a rank-``rank`` SVD to the filled matrix with
    singular values shrunk by ``(lambda_s - sigma^2) / lambda_s`` (sigma^2 = mean of the discarded eigenvalues)
    and refills the missing cells.
    """

    def __init__(self, rank: int = 5, max_iter: int = 500, tol: float = 1e-6):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def _complete(self, data: ModelData) -> np.ndarray:
        Zt = _train(data)
        mask = data.mask
        n, p = Zt.shape
        r = min(self.rank, min(n, p) - 1)
        Y = np.where(mask, Zt, _col_means(Zt))
        fit = Y
        for _ in range(self.max_iter):
            mu = Y.mean(axis=0)
            U, s, Vt = np.linalg.svd(Y - mu, full_matrices=False)
            eig = s**2 / n
            sigma2 = eig[r:].mean() if eig.size > r else 0.0
            shrink = np.clip((eig[:r] - sigma2) / np.where(eig[:r] > 0, eig[:r], 1.0), 0.0, 1.0)
            fit = (U[:, :r] * (s[:r] * shrink)) @ Vt[:r] + mu
            new = np.where(mask, Zt, fit)
            change = np.sum((new - Y) ** 2) / (np.sum(Y**2) + 1e-12)
            Y = new
            if change < self.tol:
                break
        return fit


class MissForest(_Baseline):
    """Chained random forests (Stekhoven and Bühlmann, 2012) via scikit-learn's ``IterativeImputer``.

    The level of each row enters as a one-hot auxiliary feature. Predictions at observed cells are the
    out-of-bag predictions of the final round's forests.
    """

    def __init__(self, n_estimators: int = 50, max_iter: int = 5, seed: int = 0, n_jobs: int = 1):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.seed = seed
        self.n_jobs = n_jobs

    def _complete(self, data: ModelData) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        Zt = _train(data)
        level = data.levels()
        aux = (level[:, None] == np.unique(level)[None, :]).astype(float)
        empty = ~np.isfinite(Zt).any(axis=0)
        X = np.hstack([np.where(empty, 0.0, Zt), aux])
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            oob_score=True,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        imp = IterativeImputer(
            estimator=rf,
            max_iter=self.max_iter,
            initial_strategy="mean",
            random_state=self.seed,
            skip_complete=True,
        )
        P = imp.fit_transform(X)[:, : Zt.shape[1]]
        last: dict[int, object] = {}
        for triplet in imp.imputation_sequence_:
            last[triplet.feat_idx] = triplet.estimator
        for j, est in last.items():
            obs = np.isfinite(Zt[:, j]) if j < Zt.shape[1] else None
            oob = getattr(est, "oob_prediction_", None)
            if obs is not None and oob is not None and len(oob) == obs.sum():
                P[obs, j] = np.where(np.isfinite(oob), oob, P[obs, j])
        return np.where(empty, 0.0, P)


BASELINES = {
    "GlobalMean": GlobalMean,
    "LevelMean": LevelMean,
    "ParentCarryDown": ParentCarryDown,
    "KNNGower": KNNGower,
    "SoftImpute": SoftImpute,
    "IterativePCA": IterativePCA,
    "MissForest": MissForest,
}
