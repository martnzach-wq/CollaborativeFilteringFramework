"""HierTypedMF: hierarchical, typed low-rank model (design §4, milestone M3).

    zhat_gk = mu_k + b_g + u_g . v_k

    objective = (1/N) sum_obs l_k(z_gk, zhat_gk)                               typed losses (GLRM)
              + (lam_agg/N) sum_(p,k obs) (z_pk - T_k(agg_k{T_k^-1(zhat_ck)}))^2  aggregation, original scale
              + (lam_u/n) sum_g ||u_g - sum_p m_gp u_p - A^T x_g||^2              hierarchical prior
              + (lam_v/L) ||V||^2 + (lam_b/n) ||b||^2 + (lam_a/n) ||A||^2
              + (lam_lap/n) sum_(g~h) ||u_g - u_h||^2                               spatial Laplacian
              + (lam_time/n) sum_(t, t+1) ||u_(g,t) - u_(g,t+1)||^2                 random walk (geo-time)

with N observed cells, n entities, L latent columns and membership weights m_gp. Losses: squared or Huber for
continuous columns, cumulative logit for ordinal columns, multinomial for categorical columns (one latent column
per category). The MAP estimate is found by full-batch L-BFGS in float64 (PyTorch, decision D6).

Uncertainty: block-diagonal Laplace approximation per entity over (u_g, b_g), with Gaussian likelihood
precision 1/sigma_k^2 (training residual variance per column) and prior precision implied by the penalties.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse

from regsim.models.base import ModelData
from regsim.schema import Aggregation, DType

COUPLINGS = ("none", "broadcast", "prior_only", "prior_and_constraint")
GAUSS, ORDINAL, CATEGORICAL = "gauss", "ordinal", "categorical"
COVER_MIN = 0.99
EPS = 1e-9


@dataclass
class _Layout:
    kinds: list[str]
    start: list[int]
    width: list[int]
    n_latent: int


class HierTypedMF:
    """Hierarchical typed matrix factorisation.

    Parameters
    ----------
    rank : int
        Number of latent factors.
    lam_u, lam_v, lam_b, lam_a : float
        Penalties of the entity prior, indicator factors, entity biases and side-information weights.
    lam_agg : float
        Weight of the aggregation constraints (``coupling="prior_and_constraint"``).
    lam_lap : float
        Weight of the spatial Laplacian penalty (needs ``ModelData.adjacency``).
    lam_time : float
        Weight of the random-walk penalty between consecutive windows (needs ``ModelData.time_pairs``).
    coupling : {"none", "broadcast", "prior_only", "prior_and_constraint"}
        ``none``: ridge prior on u; ``broadcast``: ancestor values copied into missing child cells and treated as
        observed (LEVITATE), ridge prior; ``prior_only``: hierarchical prior; ``prior_and_constraint``: prior and
        aggregation constraints.
    loss : {"squared", "huber"}
        Loss of continuous columns.
    huber_delta : float
        Transition point of the Huber loss (model scale).
    max_iter : int
        Maximum number of L-BFGS iterations.
    reconcile : bool
        Make predictions of aggregable columns coherent with observed or reconciled parent values after fitting.
    seed : int
        Seed of the initialisation.
    """

    def __init__(
        self,
        rank: int = 12,
        lam_u: float = 0.03,
        lam_v: float = 0.1,
        lam_b: float = 1.0,
        lam_agg: float = 1.0,
        lam_a: float = 1.0,
        lam_lap: float = 0.0,
        lam_time: float = 1.0,
        coupling: str = "prior_and_constraint",
        loss: str = "squared",
        huber_delta: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-10,
        reconcile: bool = True,
        seed: int = 0,
    ):
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}")
        if loss not in ("squared", "huber"):
            raise ValueError("loss must be 'squared' or 'huber'")
        self.rank = rank
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.lam_b = lam_b
        self.lam_agg = lam_agg
        self.lam_a = lam_a
        self.lam_lap = lam_lap
        self.lam_time = lam_time
        self.coupling = coupling
        self.loss = loss
        self.huber_delta = huber_delta
        self.max_iter = max_iter
        self.tol = tol
        self.reconcile = reconcile
        self.seed = seed

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    # ================================================================================================ fit
    def fit(self, data: ModelData) -> HierTypedMF:
        self.data_ = data
        Z, mask = data.Z, data.mask
        if self.coupling == "broadcast":
            Z, mask = broadcast_fill(data)
        self.layout_ = _layout(data)
        n, p = Z.shape
        lay = self.layout_
        r = self.rank
        dt = torch.float64
        gen = torch.Generator().manual_seed(self.seed)

        Zt = torch.tensor(np.where(mask, Z, 0.0), dtype=dt)
        Mt = torch.tensor(mask, dtype=dt)
        n_obs = max(float(mask.sum()), 1.0)
        self.n_obs_ = n_obs
        gauss = [j for j, k in enumerate(lay.kinds) if k == GAUSS]
        self.gauss_ = gauss

        # Initialisation: column means and a scaled truncated SVD of the mean-filled continuous columns.
        mu0 = np.zeros(lay.n_latent)
        for j in range(p):
            col = Z[mask[:, j], j]
            if lay.kinds[j] == GAUSS and col.size:
                mu0[lay.start[j]] = col.mean()
        U0 = 0.1 * torch.randn(n, r, generator=gen, dtype=dt)
        V0 = 0.1 * torch.randn(lay.n_latent, r, generator=gen, dtype=dt)
        if gauss:
            G = np.where(mask[:, gauss], Z[:, gauss] - mu0[[lay.start[j] for j in gauss]], 0.0)
            u, s, vt = np.linalg.svd(G, full_matrices=False)
            k = min(r, s.size)
            root = np.sqrt(s[:k] / max(mask[:, gauss].mean(), 1e-3))  # split singular values symmetrically
            U0[:, :k] = torch.tensor(u[:, :k] * root, dtype=dt)
            V0[[lay.start[j] for j in gauss], :k] = torch.tensor(vt[:k].T * root, dtype=dt)

        mu = torch.tensor(mu0, dtype=dt, requires_grad=True)
        b = torch.zeros(n, dtype=dt, requires_grad=True)
        # U is optimised as alpha * U_t, alpha balancing the per-element penalties of U and V (preconditioning).
        alpha = float(np.clip(np.sqrt((self.lam_v / lay.n_latent) / max(self.lam_u / n, 1e-12)), 1e-3, 1e3))
        U_t = (U0 / alpha).clone().requires_grad_(True)
        V = V0.clone().requires_grad_(True)
        params = [mu, b, U_t, V]
        X = None
        A = None
        if data.side is not None and data.side.size:
            X = torch.tensor(data.side, dtype=dt)
            A = torch.zeros(X.shape[1], r, dtype=dt, requires_grad=True)
            params.append(A)
        thetas = {}
        for j in range(p):
            if lay.kinds[j] == ORDINAL:
                c = lay.width[j]  # number of categories (one latent column, c - 1 thresholds)
                codes = Z[mask[:, j], j]
                freq = (
                    np.array([(codes <= q).mean() for q in range(c - 1)])
                    if codes.size
                    else np.linspace(0.2, 0.8, c - 1)
                )
                th = np.log(np.clip(freq, 0.02, 0.98) / (1 - np.clip(freq, 0.02, 0.98)))
                th = np.maximum.accumulate(th + np.arange(c - 1) * 1e-3)
                raw = np.concatenate([[th[0]], np.log(np.expm1(np.maximum(np.diff(th), 1e-3)))])
                thetas[j] = torch.tensor(raw, dtype=dt, requires_grad=True)
                params.append(thetas[j])

        prior = self.coupling in ("prior_only", "prior_and_constraint")
        P = _parent_matrix(data.parent, n) if prior else None
        agg_terms = self._agg_terms(data, mask) if self.coupling == "prior_and_constraint" else []
        lap = _pairs(data.adjacency) if (self.lam_lap > 0 and data.adjacency is not None) else None
        tpairs = data.time_pairs if (data.time_pairs is not None and len(data.time_pairs)) else None
        typed = self._typed_targets(Z, mask)
        delta = self.huber_delta

        def objective():
            U = alpha * U_t
            Zh = mu + b[:, None] + U @ V.T
            loss = Zt.new_zeros(())
            if gauss:
                cols = [lay.start[j] for j in gauss]
                res = (Zh[:, cols] - Zt[:, gauss]) * Mt[:, gauss]
                if self.loss == "squared":
                    loss = loss + (res**2).sum()
                else:
                    loss = loss + 2 * torch.nn.functional.huber_loss(
                        res, torch.zeros_like(res), delta=delta, reduction="sum"
                    )
            for j, (rows, codes) in typed.items():
                if lay.kinds[j] == ORDINAL:
                    loss = loss - _ordinal_loglik(Zh[rows, lay.start[j]], codes, _thresholds(thetas[j])).sum()
                else:
                    logits = Zh[rows, lay.start[j] : lay.start[j] + lay.width[j]]
                    loss = loss + torch.nn.functional.cross_entropy(logits, codes, reduction="sum")
            for term in agg_terms:
                child = _inverse(Zh[:, term["col"]] * term["scale"] + term["loc"], term["kind"])
                agg = torch.sparse.mm(term["W"], child[:, None])[:, 0][term["rows"]]
                zp = (_forward(agg, term["kind"]) - term["loc"]) / term["scale"]
                loss = loss + self.lam_agg * ((term["target"] - zp) ** 2).sum()
            obj = loss / n_obs
            mean = torch.sparse.mm(P, U) if P is not None else torch.zeros_like(U)
            if A is not None:
                mean = mean + X @ A
                obj = obj + self.lam_a * (A**2).sum() / n
            obj = obj + self.lam_u * ((U - mean) ** 2).sum() / n
            obj = obj + self.lam_v * (V**2).sum() / lay.n_latent + self.lam_b * (b**2).sum() / n
            if lap is not None:
                obj = obj + self.lam_lap * ((U[lap[0]] - U[lap[1]]) ** 2).sum() / n
            if tpairs is not None:
                obj = obj + self.lam_time * ((U[tpairs[:, 0]] - U[tpairs[:, 1]]) ** 2).sum() / n
            return obj

        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=self.tol,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            obj = objective()
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            self.objective_ = float(objective())
        self.n_iter_ = opt.state[opt._params[0]].get("n_iter", 0)
        if self.n_iter_ >= self.max_iter:
            warnings.warn(f"HierTypedMF reached max_iter={self.max_iter}", stacklevel=2)

        self.mu_ = mu.detach().numpy().copy()
        self.b_ = b.detach().numpy().copy()
        self.U_ = (alpha * U_t).detach().numpy().copy()
        self.V_ = V.detach().numpy().copy()
        self.A_ = None if A is None else A.detach().numpy().copy()
        self.thresholds_ = {j: _thresholds(t).detach().numpy().copy() for j, t in thetas.items()}
        self.latent_ = self.mu_ + self.b_[:, None] + self.U_ @ self.V_.T
        self._noise(data)
        self._laplace(data)
        self.P_ = self._outputs(self.latent_)
        if self.reconcile:
            self.P_ = reconcile(data, self.P_, self.predict_std() ** 2)
        return self

    # ================================================================================================ outputs
    def predict(self) -> np.ndarray:
        """Point predictions on the model scale (expected code for ordinal, mode for categorical columns)."""
        return self.P_

    def predict_latent(self) -> np.ndarray:
        return self.latent_

    def predict_proba(self, column: int) -> np.ndarray:
        """Category probabilities of an ordinal or categorical column, shape ``(n, n_categories)``."""
        return self._proba(self.latent_, column)

    def predict_std(self) -> np.ndarray:
        """Predictive standard deviation of continuous columns (NaN elsewhere): Laplace variance plus noise."""
        out = np.full(self.P_.shape, np.nan)
        lay = self.layout_
        for j in self.gauss_:
            phi = np.append(self.V_[lay.start[j]], 1.0)
            var = np.einsum("i,nij,j->n", phi, self.cov_, phi) + self.sigma2_[j]
            out[:, j] = np.sqrt(var)
        return out

    def predict_dist(
        self, n_samples: int = 200, seed: int | np.random.Generator = 0, noise: bool = True
    ) -> np.ndarray:
        """Samples ``(n_samples, n, p)``: entity parameters from the Laplace posterior plus observation noise
        (``noise=False``: expected values only); ordinal and categorical cells are drawn from their predictive
        probabilities."""
        rng = np.random.default_rng(seed)
        n, p = self.P_.shape
        lay = self.layout_
        L = np.linalg.cholesky(self.cov_ + 1e-12 * np.eye(self.cov_.shape[1]))
        out = np.empty((n_samples, n, p))
        for s in range(n_samples):
            theta = np.einsum("nij,nj->ni", L, rng.standard_normal((n, self.rank + 1)))
            lat = self.mu_ + (self.b_ + theta[:, -1])[:, None] + (self.U_ + theta[:, :-1]) @ self.V_.T
            for j in range(p):
                if lay.kinds[j] == GAUSS:
                    eps = np.sqrt(self.sigma2_[j]) * rng.standard_normal(n) if noise else 0.0
                    out[s, :, j] = lat[:, lay.start[j]] + eps
                else:
                    pr = self._proba(lat, j)
                    u = rng.random(n)[:, None]
                    out[s, :, j] = (u > np.cumsum(pr, axis=1)).sum(axis=1)
        return out

    def embed_entities(self) -> np.ndarray:
        return self.U_

    def embed_indicators(self) -> np.ndarray:
        lay = self.layout_
        return np.vstack(
            [
                self.V_[
                    lay.start[j] : lay.start[j] + (1 if lay.kinds[j] != CATEGORICAL else lay.width[j])
                ].mean(axis=0)
                for j in range(len(lay.kinds))
            ]
        )

    # ================================================================================================ fold-in
    def fold_in(
        self, z_new, mask_new, parent_new=None, side_new=None, rows=None, n_steps: int = 2
    ) -> np.ndarray:
        """Predictions for new entities from their observed continuous cells, parents and side information.

        Solves the penalised least-squares problem for (u, b) with the indicator side fixed (design §7, P2).
        With ``rows`` (the entities' rows in the fitted data, whose own cells were not used for fitting) and
        ``coupling="prior_and_constraint"``, the observed values of their parents enter as linearised
        aggregation pseudo-observations with the siblings' values fixed (``n_steps`` Gauss-Newton steps).
        The posterior covariance of the new entities is kept in ``fold_in_cov_``.
        """
        z_new = np.atleast_2d(np.asarray(z_new, dtype=float))
        mask_new = np.atleast_2d(np.asarray(mask_new, dtype=bool))
        m = z_new.shape[0]
        parent_new = parent_new or [[] for _ in range(m)]
        lay = self.layout_
        r = self.rank
        n = self.U_.shape[0]
        lam_u = self.lam_u * self.n_obs_ / n
        lam_b = self.lam_b * self.n_obs_ / n
        out_lat = np.empty((m, lay.n_latent))
        covs = np.empty((m, r + 1, r + 1))
        prior_on = self.coupling in ("prior_only", "prior_and_constraint")
        use_agg = rows is not None and self.coupling == "prior_and_constraint" and self.lam_agg > 0
        agg_cols = self._fold_in_agg_columns() if use_agg else []
        for i in range(m):
            prior = np.zeros(r)
            if prior_on:
                for pidx, w in parent_new[i]:
                    prior += w * self.U_[pidx]
            if self.A_ is not None and side_new is not None:
                prior += np.asarray(side_new[i]) @ self.A_
            cols = [j for j in self.gauss_ if mask_new[i, j]]
            Phi = np.column_stack([self.V_[[lay.start[j] for j in cols]], np.ones(len(cols))])
            y = z_new[i, cols] - self.mu_[[lay.start[j] for j in cols]]
            reg = np.diag(np.r_[np.full(r, lam_u), lam_b])
            rhs = Phi.T @ y + np.r_[lam_u * prior, 0.0]
            if rows is not None and prior_on:
                # Children in the fitted data: their factors inform u through their prior.
                for c, w in self._children().get(int(rows[i]), []):
                    reg[:r, :r] += lam_u * w**2 * np.eye(r)
                    rhs[:r] += lam_u * w * self.U_[c]
            if use_agg:
                Ac, tc = self._agg_from_children(int(rows[i]), agg_cols, mask_new[i])
                if Ac:
                    Ac, tc = np.array(Ac), np.array(tc)
                    reg = reg + self.lam_agg * Ac.T @ Ac
                    rhs = rhs + self.lam_agg * Ac.T @ tc
            H = Phi.T @ Phi + reg
            theta = np.linalg.solve(H, rhs)
            if use_agg:
                for _ in range(n_steps):
                    A_rows, t_rows = self._agg_pseudo(int(rows[i]), theta, agg_cols, mask_new[i])
                    if not A_rows:
                        break
                    Aa, ta = np.array(A_rows), np.array(t_rows)
                    H = Phi.T @ Phi + reg + self.lam_agg * Aa.T @ Aa
                    theta = np.linalg.solve(H, rhs + self.lam_agg * Aa.T @ ta)
            out_lat[i] = self.mu_ + theta[-1] + theta[:-1] @ self.V_.T
            s2 = np.mean([self.sigma2_[j] for j in cols]) if cols else np.mean(list(self.sigma2_.values()))
            covs[i] = np.linalg.inv(H) * s2
        self.fold_in_cov_ = covs
        return self._outputs(out_lat)

    def _fold_in_agg_columns(self) -> list[tuple]:
        data = self.data_
        out = []
        for j in self.gauss_:
            ind = data.column_meta[j]
            agg = getattr(ind, "aggregation", Aggregation.NONE)
            W = _agg_matrix(data, agg, getattr(ind, "ratio_of", None))
            if W is None:
                continue
            sc = data.scalers[j] if data.scalers else ("identity", 0.0, 1.0)
            out.append((j, agg, sparse.csr_matrix(W), sparse.csc_matrix(W), sc))
        return out

    def _children(self) -> dict[int, list[tuple[int, float]]]:
        if "children_" not in self.__dict__:
            out: dict[int, list[tuple[int, float]]] = {}
            for c, ps in enumerate(self.data_.parent):
                for p, w in ps:
                    out.setdefault(p, []).append((c, w))
            self.children_ = out
        return self.children_

    def _agg_from_children(self, g: int, agg_cols: list[tuple], own_mask: np.ndarray):
        """Rows ``phi_j . theta ~ T(aggregate of children) - mu_j`` for unobserved cells of a parent row ``g``."""
        data = self.data_
        lay = self.layout_
        A_rows, t_rows = [], []
        for j, agg, Wr, _, sc in agg_cols:
            if own_mask[j]:
                continue
            kids = Wr.indices[Wr.indptr[g] : Wr.indptr[g + 1]]
            if kids.size == 0:
                continue
            if agg is Aggregation.SUM and data.agg_cover is not None and data.agg_cover[g] < COVER_MIN:
                continue
            w = Wr.data[Wr.indptr[g] : Wr.indptr[g + 1]]
            zc = np.where(data.mask[kids, j], data.Z[kids, j], self.latent_[kids, lay.start[j]])
            kind, loc, scale = sc
            target = (_np_forward(np.array([np.sum(w * _to_orig(zc, sc))]), kind)[0] - loc) / scale
            A_rows.append(np.append(self.V_[lay.start[j]], 1.0))
            t_rows.append(target - self.mu_[lay.start[j]])
        return A_rows, t_rows

    def _agg_pseudo(self, g: int, theta: np.ndarray, agg_cols: list[tuple], own_mask: np.ndarray):
        """Linearised aggregation rows for entity row ``g``: parent value ~ aggregate with siblings fixed."""
        data = self.data_
        lay = self.layout_
        A_rows, t_rows = [], []
        for j, agg, Wr, Wc, sc in agg_cols:
            if own_mask[j]:
                continue
            col = Wc[:, g]
            for p, w_g in zip(col.indices, col.data, strict=True):
                if not data.mask[p, j]:
                    continue
                if agg is Aggregation.SUM and data.agg_cover is not None and data.agg_cover[p] < COVER_MIN:
                    continue
                kids = Wr.indices[Wr.indptr[p] : Wr.indptr[p + 1]]
                w = Wr.data[Wr.indptr[p] : Wr.indptr[p + 1]]
                others = kids != g
                zc = np.where(data.mask[kids, j], data.Z[kids, j], self.latent_[kids, lay.start[j]])
                phi = np.append(self.V_[lay.start[j]], 1.0)
                z0 = self.mu_[lay.start[j]] + phi @ theta
                S0 = np.sum(w[others] * _to_orig(zc[others], sc)) + w_g * _to_orig(np.array([z0]), sc)[0]
                kind, loc, scale = sc
                zS0 = (_np_forward(np.array([S0]), kind)[0] - loc) / scale
                a = _forward_slope(S0, kind) / scale * w_g * _derivative(np.array([z0]), sc)[0]
                A_rows.append(a * phi)
                t_rows.append(data.Z[p, j] - zS0 + a * (phi @ theta))
        return A_rows, t_rows

    def fold_in_indicator(self, z_col, mask_col) -> np.ndarray:
        """Predictions for a new continuous indicator observed on some entities (design §7, P3)."""
        z_col = np.asarray(z_col, dtype=float)
        mask_col = np.asarray(mask_col, dtype=bool)
        r = self.rank
        rows = np.flatnonzero(mask_col)
        Phi = np.column_stack([self.U_[rows], np.ones(rows.size)])
        y = z_col[rows] - self.b_[rows]
        lam_v = self.lam_v * self.n_obs_ / self.layout_.n_latent
        reg = np.diag(np.r_[np.full(r, lam_v), 0.0])
        theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
        return theta[-1] + self.b_ + self.U_ @ theta[:-1]

    # ================================================================================================ internals
    def _outputs(self, latent: np.ndarray) -> np.ndarray:
        lay = self.layout_
        out = np.empty((latent.shape[0], len(lay.kinds)))
        for j, kind in enumerate(lay.kinds):
            if kind == GAUSS:
                out[:, j] = latent[:, lay.start[j]]
            else:
                pr = self._proba(latent, j)
                out[:, j] = pr @ np.arange(pr.shape[1]) if kind == ORDINAL else pr.argmax(axis=1)
        return out

    def _proba(self, latent: np.ndarray, j: int) -> np.ndarray:
        lay = self.layout_
        if lay.kinds[j] == ORDINAL:
            th = self.thresholds_[j]
            cdf = 1 / (1 + np.exp(-(th[None, :] - latent[:, lay.start[j], None])))
            cdf = np.hstack([np.zeros((latent.shape[0], 1)), cdf, np.ones((latent.shape[0], 1))])
            return np.clip(np.diff(cdf, axis=1), 0.0, 1.0)
        if lay.kinds[j] == CATEGORICAL:
            z = latent[:, lay.start[j] : lay.start[j] + lay.width[j]]
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        raise ValueError(f"column {j} is continuous")

    def _typed_targets(self, Z: np.ndarray, mask: np.ndarray) -> dict:
        out = {}
        for j, kind in enumerate(self.layout_.kinds):
            if kind == GAUSS:
                continue
            rows = np.flatnonzero(mask[:, j])
            codes = np.clip(np.rint(Z[rows, j]).astype(int), 0, self.layout_.width[j] - 1)
            out[j] = (torch.tensor(rows), torch.tensor(codes))
        return out

    def _agg_terms(self, data: ModelData, mask: np.ndarray) -> list[dict]:
        terms = []
        for j in self.gauss_:
            ind = data.column_meta[j]
            agg = getattr(ind, "aggregation", Aggregation.NONE)
            W = _agg_matrix(data, agg, getattr(ind, "ratio_of", None))
            if W is None:
                continue
            has_kids = np.asarray(W.sum(axis=1)).ravel() > 0
            rows = has_kids & mask[:, j]
            if agg is Aggregation.SUM and data.agg_cover is not None:
                rows &= data.agg_cover >= COVER_MIN
            rows = np.flatnonzero(rows)
            if rows.size == 0:
                continue
            kind, loc, scale = data.scalers[j] if data.scalers else ("identity", 0.0, 1.0)
            terms.append(
                {
                    "col": self.layout_.start[j],
                    "W": _torch_sparse(W),
                    "rows": torch.tensor(rows),
                    "target": torch.tensor(data.Z[rows, j], dtype=torch.float64),
                    "kind": kind,
                    "loc": float(loc),
                    "scale": float(scale),
                }
            )
        return terms

    def _noise(self, data: ModelData) -> None:
        self.sigma2_ = {}
        for j in self.gauss_:
            m = data.mask[:, j]
            res = self.latent_[m, self.layout_.start[j]] - data.Z[m, j]
            self.sigma2_[j] = float(max(np.mean(res**2), 1e-4)) if res.size else 1.0

    def _laplace(self, data: ModelData) -> None:
        n = self.U_.shape[0]
        r = self.rank
        lay = self.layout_
        s2bar = float(np.mean(list(self.sigma2_.values()))) if self.sigma2_ else 1.0
        tau_u = self.lam_u * self.n_obs_ / (n * s2bar)
        tau_b = self.lam_b * self.n_obs_ / (n * s2bar)
        child_w = np.zeros(n)
        if self.coupling in ("prior_only", "prior_and_constraint"):
            for ps in data.parent:
                for p, w in ps:
                    child_w[p] += w**2
        Phi = np.column_stack([self.V_[[lay.start[j] for j in self.gauss_]], np.ones(len(self.gauss_))])
        prec = np.array([1.0 / self.sigma2_[j] for j in self.gauss_])
        M = data.mask[:, self.gauss_].astype(float) * prec[None, :]
        # H_g = sum_k m_gk phi_k phi_k^T / sigma_k^2 + prior precision
        H = np.einsum("nk,ki,kj->nij", M, Phi, Phi)
        H[:, np.arange(r), np.arange(r)] += tau_u * (1.0 + child_w)[:, None]
        H[:, r, r] += tau_b
        self.cov_ = np.linalg.inv(H)


# ==================================================================================================== helpers
def _layout(data: ModelData) -> _Layout:
    kinds, start, width = [], [], []
    pos = 0
    for j, ind in enumerate(data.column_meta):
        dtype = getattr(ind, "dtype", DType.CONTINUOUS)
        if dtype is DType.ORDINAL:
            c = len(ind.categories) or int(np.nanmax(data.Z[:, j])) + 1
            kinds.append(ORDINAL)
            start.append(pos)
            width.append(max(c, 2))
            pos += 1
        elif dtype is DType.CATEGORICAL:
            c = len(ind.categories) or int(np.nanmax(data.Z[:, j])) + 1
            kinds.append(CATEGORICAL)
            start.append(pos)
            width.append(max(c, 2))
            pos += max(c, 2)
        else:
            kinds.append(GAUSS)
            start.append(pos)
            width.append(1)
            pos += 1
    if not kinds:  # no metadata: all continuous
        p = data.Z.shape[1]
        return _Layout([GAUSS] * p, list(range(p)), [1] * p, p)
    return _Layout(kinds, start, width, pos)


def _thresholds(raw: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(torch.cat([raw[:1], torch.nn.functional.softplus(raw[1:])]), dim=0)


def _ordinal_loglik(z: torch.Tensor, codes: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    inf = torch.tensor([1e6], dtype=th.dtype)
    ext = torch.cat([-inf, th, inf])
    upper = torch.sigmoid(ext[codes + 1] - z)
    lower = torch.sigmoid(ext[codes] - z)
    return torch.log(torch.clamp(upper - lower, min=1e-12))


def _forward(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in ("identity", "rank"):
        return x
    if kind == "log":
        return torch.log(torch.clamp(x, min=EPS))
    if kind == "log1p":
        return torch.log1p(torch.clamp(x, min=-1 + EPS))
    if kind == "logit":
        x = torch.clamp(x, EPS, 1 - EPS)
        return torch.log(x / (1 - x))
    raise ValueError(f"unsupported transform {kind}")


def _inverse(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in ("identity", "rank"):
        return z
    if kind == "log":
        return torch.exp(torch.clamp(z, max=50))
    if kind == "log1p":
        return torch.expm1(torch.clamp(z, max=50))
    if kind == "logit":
        return torch.sigmoid(z)
    raise ValueError(f"unsupported transform {kind}")


def _agg_matrix(data: ModelData, agg: Aggregation, ratio_of: tuple[str, str] | None = None):
    """Aggregation matrix of a column: population or area weights for means, membership for sums.

    A ratio is aggregated as the ratio of sums, i.e. a mean weighted by its denominator (area for
    ``ratio_of=(..., "area_km2")``, population otherwise).
    """
    if agg is Aggregation.RATIO and ratio_of and ratio_of[1] in ("area_km2", "area"):
        return data.agg_area if data.agg_area is not None else data.agg
    if agg in (Aggregation.WMEAN_POP, Aggregation.RATIO):
        return data.agg
    if agg is Aggregation.WMEAN_AREA:
        return data.agg_area if data.agg_area is not None else data.agg
    if agg is Aggregation.SUM:
        return data.agg_sum
    return None


def _torch_sparse(W) -> torch.Tensor:
    W = sparse.coo_matrix(W)
    idx = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(
        idx, torch.tensor(W.data, dtype=torch.float64), W.shape, check_invariants=False
    ).coalesce()


def _parent_matrix(parent: list[list[tuple[int, float]]], n: int) -> torch.Tensor:
    rows, cols, vals = [], [], []
    for c, ps in enumerate(parent):
        for p, w in ps:
            rows.append(c)
            cols.append(p)
            vals.append(w)
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return _torch_sparse(W)


def _pairs(adjacency) -> tuple[torch.Tensor, torch.Tensor] | None:
    A = sparse.triu(sparse.coo_matrix(adjacency), k=1).tocoo()
    if A.nnz == 0:
        return None
    return torch.tensor(A.row, dtype=torch.long), torch.tensor(A.col, dtype=torch.long)


def broadcast_fill(data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    """LEVITATE mode: copy the nearest ancestor value into missing child cells and treat it as observed."""
    Z = np.where(data.mask, data.Z, np.nan)
    level = data.levels()
    for g in np.argsort(level, kind="stable"):
        miss = np.isnan(Z[g])
        if not miss.any() or not data.parent[g]:
            continue
        num = np.zeros(Z.shape[1])
        den = np.zeros(Z.shape[1])
        for p, w in data.parent[g]:
            ok = np.isfinite(Z[p])
            num[ok] += w * Z[p, ok]
            den[ok] += w
        fill = miss & (den > 0)
        Z[g, fill] = num[fill] / den[fill]
    mask = np.isfinite(Z)
    return np.where(mask, Z, 0.0), mask


def reconcile(
    data: ModelData,
    P: np.ndarray,
    var: np.ndarray | None = None,
    obs_var: float = 0.05**2,
    n_steps: int = 3,
) -> np.ndarray:
    """Make predictions of aggregable continuous columns coherent across levels.

    For each column, the constraints "parent = aggregate of its children" (original scale) are linearised on
    the model scale and solved jointly by a variance-weighted least-change projection (MinT-type with diagonal
    weights; Wickramasuriya et al., 2019), repeated ``n_steps`` times (Gauss-Newton). Predicted cells move with
    their predictive variance ``var``; observed cells enter with the measurement variance ``obs_var`` so that
    inconsistencies between published values are attributed to the data, but only predicted cells are changed.
    Constraints are expressed relative to the parent value, which keeps the system well conditioned.

    Parameters
    ----------
    data : ModelData
        Training data (``mask`` marks observed cells).
    P : ndarray
        Predictions on the model scale.
    var : ndarray, optional
        Predictive variance on the model scale (default 1).
    obs_var : float
        Measurement variance of observed cells on the model scale.
    n_steps : int
        Gauss-Newton steps.
    """
    from scipy.sparse.linalg import spsolve

    P = P.copy()
    for j, ind in enumerate(data.column_meta):
        agg = getattr(ind, "aggregation", Aggregation.NONE)
        if getattr(ind, "dtype", DType.CONTINUOUS) in (DType.ORDINAL, DType.CATEGORICAL):
            continue
        W = _agg_matrix(data, agg, getattr(ind, "ratio_of", None))
        if W is None:
            continue
        W = sparse.csr_matrix(W)
        sc = data.scalers[j] if data.scalers else ("identity", 0.0, 1.0)
        free = ~data.mask[:, j]
        parents = np.flatnonzero(np.diff(W.indptr) > 0)
        if agg is Aggregation.SUM and data.agg_cover is not None:
            parents = parents[data.agg_cover[parents] >= COVER_MIN]
        C = (
            W[parents]
            - sparse.csr_matrix(
                (np.ones(parents.size), (np.arange(parents.size), parents)), shape=(parents.size, W.shape[1])
            )
        ).tocsr()
        involved = np.flatnonzero(np.asarray(abs(C)[:, free].sum(axis=1)).ravel() > 0)
        if involved.size == 0:
            continue
        C = C[involved]
        cells = np.flatnonzero(np.asarray(abs(C).sum(axis=0)).ravel() > 0)
        C = C[:, cells].tocsr()
        is_free = free[cells]
        v = np.where(is_free, 1.0 if var is None else np.nan_to_num(var[cells, j], nan=1.0), obs_var)
        v = np.maximum(v, 1e-12)
        z = np.where(data.mask[:, j], data.Z[:, j], P[:, j])[cells]
        for _ in range(n_steps):
            x = _to_orig(z, sc)
            resid = -(C @ x)
            norm = np.abs(sparse.csr_matrix(abs(C)) @ np.abs(x)) + 1e-12  # relative constraints
            Cn = sparse.diags(1.0 / norm) @ C
            CJ = Cn @ sparse.diags(_derivative(z, sc))
            M = (CJ @ sparse.diags(v) @ CJ.T).tocsc()
            M = M + 1e-9 * sparse.identity(M.shape[0], format="csc") * max(M.diagonal().mean(), 1e-12)
            lam = spsolve(M, resid / norm)
            dz = v * (CJ.T @ lam)
            z = z + np.where(is_free, dz, 0.0)
        P[cells[is_free], j] = z[is_free]
    return P


def _forward_slope(x: float, kind: str) -> float:
    """d T(x) / d x of the transform at x."""
    if kind == "log":
        return 1.0 / max(x, EPS)
    if kind == "log1p":
        return 1.0 / max(1.0 + x, EPS)
    if kind == "logit":
        x = min(max(x, EPS), 1 - EPS)
        return 1.0 / (x * (1 - x))
    return 1.0


def _derivative(z: np.ndarray, sc) -> np.ndarray:
    """d x / d z of the map from model scale to original scale."""
    kind, loc, scale = sc
    x = _np_inverse(z * scale + loc, kind)
    if kind == "log":
        return x * scale
    if kind == "log1p":
        return (x + 1) * scale
    if kind == "logit":
        return x * (1 - x) * scale
    return np.full_like(z, scale, dtype=float)


def coherence_error(data: ModelData, P: np.ndarray, only_predicted: bool = True) -> np.ndarray:
    """Relative gap |aggregate of children - parent| / |parent| on the original scale, per (parent, column) with an
    observed parent value; children take observed values where available and predictions elsewhere."""
    out = []
    for j, ind in enumerate(data.column_meta):
        agg = getattr(ind, "aggregation", Aggregation.NONE)
        if getattr(ind, "dtype", DType.CONTINUOUS) in (DType.ORDINAL, DType.CATEGORICAL):
            continue
        W = _agg_matrix(data, agg, getattr(ind, "ratio_of", None))
        if W is None:
            continue
        W = sparse.csr_matrix(W)
        kind, loc, scale = data.scalers[j] if data.scalers else ("identity", 0.0, 1.0)
        vals = np.where(data.mask[:, j], data.Z[:, j], P[:, j])
        x = _np_inverse(vals * scale + loc, kind)
        for p in range(W.shape[0]):
            start, end = W.indptr[p], W.indptr[p + 1]
            if start == end or not data.mask[p, j]:
                continue
            if agg is Aggregation.SUM and data.agg_cover is not None and data.agg_cover[p] < COVER_MIN:
                continue
            kids = W.indices[start:end]
            if only_predicted and data.mask[kids, j].all():
                continue
            y = _np_inverse(data.Z[p, j] * scale + loc, kind)
            out.append(abs(np.sum(W.data[start:end] * x[kids]) - y) / max(abs(y), 1e-12))
    return np.array(out)


def _to_orig(z, sc):
    kind, loc, scale = sc
    return _np_inverse(z * scale + loc, kind)


def _to_model(x, sc):
    kind, loc, scale = sc
    return (_np_forward(x, kind) - loc) / scale


def _np_forward(x, kind):
    from regsim.prep.transforms import forward

    return forward(x, kind)


def _np_inverse(z, kind):
    from regsim.prep.transforms import inverse

    return inverse(z, kind)
