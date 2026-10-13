"""HierTypedMF (design §4, milestone M3).

z_gk = mu_k + b_g + u_g . v_k
loss = sum_obs l_k(z_gk, zhat_gk)                       typed losses (GLRM)
     + lam_u sum_g ||u_g - sum_p w_gp u_p - A x_g||^2    hierarchical prior with fractional parents
     + lam_agg sum_(p,k obs) (y_pk - agg_k({T_k^-1(zhat_ck)}))^2 / s_k^2   aggregation on original scale
     + lam_v ||V||^2 + lam_b ||b||^2 + lam_L tr(U^T L U)  optional spatial Laplacian
Posterior: Laplace approximation per entity (block-diagonal) for intervals and fold-in.
"""
from __future__ import annotations


class HierTypedMF:
    def __init__(self, rank: int = 12, lam_u: float = 0.03, lam_v: float = 0.1, lam_b: float = 1.0,
                 lam_agg: float = 1.0, coupling: str = "prior_and_constraint", seed: int = 0):
        self.params = dict(rank=rank, lam_u=lam_u, lam_v=lam_v, lam_b=lam_b, lam_agg=lam_agg,
                           coupling=coupling, seed=seed)

    def fit(self, data):
        raise NotImplementedError("M3")
