"""Feasibility prototype for regsim (reference material for the implementation, not production code).

Matrix Z (n_geo x n_ind) on the transformed and robust-scaled scale, NaN = missing.
Rows are geo-entities of several levels (N0, N2, N3), linked by `parent` (index of parent row, -1 for roots).
Aggregation weights `agg_W` (sparse-like dense matrix n x n): row p holds population shares of the children of p.
"""
import numpy as np

# ---------------------------------------------------------------- transforms
def transform(x, kind):
    if kind == "log":
        return np.log(np.clip(x, 1e-9, None))
    if kind == "log1p":
        return np.log1p(np.clip(x, 0, None))
    return x.astype(float)


def robust_scale(col):
    med = np.nanmedian(col)
    q1, q3 = np.nanpercentile(col, [25, 75])
    s = (q3 - q1) / 1.349
    return med, (s if s > 0 else (np.nanstd(col) or 1.0))


# ---------------------------------------------------------------- baselines
def pred_level_mean(Z, M, levels):
    """Mean of training cells of the same level; global mean as fallback."""
    P = np.full(Z.shape, np.nan)
    Zt = np.where(M, Z, np.nan)
    glob = np.nanmean(Zt, axis=0)
    for L in np.unique(levels):
        r = levels == L
        with np.errstate(all="ignore"):
            m = np.nanmean(Zt[r], axis=0)
        P[r] = np.where(np.isnan(m), glob, m)
    return np.where(np.isnan(P), 0.0, P)


def pred_carry_down(Z, M, levels, parent):
    """Value of the nearest ancestor with a training value (LEVITATE copying), else level mean."""
    Zt = np.where(M, Z, np.nan)
    P = pred_level_mean(Z, M, levels)
    n = Z.shape[0]
    for g in range(n):
        for k in range(Z.shape[1]):
            a = parent[g]
            while a >= 0:
                if not np.isnan(Zt[a, k]):
                    P[g, k] = Zt[a, k]
                    break
                a = parent[a]
    return P


def pred_knn_gower(Z, M, levels, eval_mask, k_nn=10):
    """D3.6-style Gower distance (min-max per column, pairwise deletion) among rows of the same level,
    inverse-distance weighted mean of the k nearest rows with the target observed."""
    Zt = np.where(M, Z, np.nan)
    rng = np.nanmax(Zt, 0) - np.nanmin(Zt, 0)
    rng[~np.isfinite(rng) | (rng == 0)] = 1.0
    X = Zt / rng
    P = pred_level_mean(Z, M, levels)
    for L in np.unique(levels):
        rows = np.where(levels == L)[0]
        XL = X[rows]
        obsL = ~np.isnan(XL)
        for qi, g in enumerate(rows):
            ks = np.where(eval_mask[g])[0]
            if len(ks) == 0:
                continue
            xq = XL[qi]
            oq = obsL[qi]
            if oq.sum() == 0:
                continue
            common = obsL & oq
            diff = np.abs(np.where(common, XL - xq, 0.0)).sum(1)
            cnt = common.sum(1)
            with np.errstate(all="ignore"):
                d = diff / cnt
            d[cnt == 0] = np.inf
            d[qi] = np.inf
            for k in ks:
                cand = np.where(obsL[:, k] & np.isfinite(d))[0]
                if len(cand) == 0:
                    continue
                nn = cand[np.argsort(d[cand])[:k_nn]]
                w = 1.0 / (d[nn] + 1e-3)
                P[g, k] = np.sum(w * Zt[rows[nn], k]) / w.sum()
    return P


def pred_softimpute(Z, M, lam=5.0, rank_max=10, iters=200):
    """SoftImpute (Mazumder et al. 2010) on column-centred data, all levels as independent rows."""
    Zt = np.where(M, Z, np.nan)
    mu = np.nanmean(Zt, 0)
    mu = np.where(np.isnan(mu), 0, mu)
    X = np.where(M, Z - mu, 0.0)
    Zh = np.zeros_like(X)
    for _ in range(iters):
        Y = np.where(M, X, Zh)
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        s = np.maximum(s - lam, 0)[:rank_max]
        Znew = (U[:, :rank_max] * s) @ Vt[:rank_max]
        if np.linalg.norm(Znew - Zh) < 1e-4 * (np.linalg.norm(Zh) + 1e-9):
            Zh = Znew
            break
        Zh = Znew
    return Zh + mu


# ---------------------------------------------------------------- hierarchical typed MF (squared loss)
def fit_hmf(Z, M, parent, agg_W=None, rank=6, lam_u=1.0, lam_v=0.1, lam_b=1.0,
            lam_agg=0.0, hier=True, iters=2500, lr=0.03, seed=0, X_side=None, lam_a=1.0):
    """MAP fit of  z_gk = mu_k + b_g + u_g.v_k  with
       hierarchical prior  u_g ~ N(u_parent(g) + A x_g, 1/lam_u)  (hier=True) or N(0, 1/lam_u),
       and aggregation constraints  z_pk ~ sum_c W_pc zhat_ck  for observed parent cells (lam_agg>0).
       Optimised with Adam on full-batch gradients."""
    rs = np.random.default_rng(seed)
    n, K = Z.shape
    Zf = np.where(M, Z, 0.0)
    Mf = M.astype(float)
    nobs = max(Mf.sum(), 1.0)
    mu = np.nanmean(np.where(M, Z, np.nan), 0)
    mu = np.where(np.isnan(mu), 0, mu)
    U = 0.1 * rs.standard_normal((n, rank))
    V = 0.1 * rs.standard_normal((K, rank))
    b = np.zeros(n)
    A = np.zeros((X_side.shape[1], rank)) if X_side is not None else None
    import scipy.sparse as sp
    rows_ = [g for g, p in enumerate(parent) if p >= 0 and hier]
    cols_ = [parent[g] for g in rows_]
    Pi = sp.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(n, n))
    if agg_W is not None and not sp.issparse(agg_W):
        agg_W = sp.csr_matrix(agg_W)
    if agg_W is not None and lam_agg > 0:
        has_child = np.asarray(agg_W.sum(1)).ravel() > 0
        Mp = (M & has_child[:, None]).astype(float)
    params = [mu, b, U, V] + ([A] if A is not None else [])
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    for t in range(1, iters + 1):
        Zh = mu + b[:, None] + U @ V.T
        G = 2 * Mf * (Zh - Zf) / nobs
        if agg_W is not None and lam_agg > 0:
            R = Mp * (agg_W @ Zh - Zf)
            G = G + 2 * lam_agg * (agg_W.T @ R) / nobs
        prior_mean = Pi @ U
        if A is not None:
            prior_mean = prior_mean + X_side @ A
        D = U - prior_mean
        gU = G @ V + 2 * lam_u * (D - (Pi.T @ D if hier else 0)) / n
        gV = G.T @ U + 2 * lam_v * V / K
        gb = G.sum(1) + 2 * lam_b * b / n
        gmu = G.sum(0)
        grads = [gmu, gb, gU, gV]
        if A is not None:
            grads.append(-2 * lam_u * X_side.T @ D / n + 2 * lam_a * A / n)
        for i, (p, g) in enumerate(zip(params, grads)):
            m1[i] = 0.9 * m1[i] + 0.1 * g
            m2[i] = 0.999 * m2[i] + 0.001 * g * g
            p -= lr * (m1[i] / (1 - 0.9 ** t)) / (np.sqrt(m2[i] / (1 - 0.999 ** t)) + 1e-8)
    return dict(mu=mu, b=b, U=U, V=V, A=A, Zhat=mu + b[:, None] + U @ V.T)


def broadcast_fill(Z, M, parent):
    """LEVITATE mode: copy the nearest ancestor value into missing child cells and treat it as observed."""
    Zb = np.where(M, Z, np.nan)
    order = np.argsort([depth(g, parent) for g in range(len(parent))])
    for g in order:
        p = parent[g]
        if p >= 0:
            miss = np.isnan(Zb[g])
            Zb[g, miss] = Zb[p, miss]
    Mb = ~np.isnan(Zb)
    return np.where(Mb, Zb, 0.0), Mb


def depth(g, parent):
    d = 0
    while parent[g] >= 0:
        g = parent[g]
        d += 1
    return d


# ---------------------------------------------------------------- similarity helpers
def whitened_profiles(Zhat, cols=None, ridge=1e-3):
    X = Zhat if cols is None else Zhat[:, cols]
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False) + ridge * np.eye(X.shape[1])
    w, Q = np.linalg.eigh(C)
    Wm = Q @ np.diag(1 / np.sqrt(w)) @ Q.T
    return Xc @ Wm


def trustworthiness_score(X_high, X_low, k=10):
    from sklearn.manifold import trustworthiness
    return trustworthiness(X_high, X_low, n_neighbors=k)
