import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from regsim.models.base import ModelData
from regsim.models.hier_mf import HierTypedMF, broadcast_fill, coherence_error, reconcile
from regsim.schema import DType, Indicator
from regsim.synthetic import generate_hierarchical

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def synth():
    d = generate_hierarchical(
        n_per_level=(8, 40, 200),
        n_indicators=15,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        seed=2,
    )
    return d, d.model_data(), d.model_data(observed_only=False).Z


def rmse(P, Z, m):
    return float(np.sqrt(np.mean((P - Z)[m] ** 2)))


def test_recovers_factor_subspaces(synth):
    d, md, _ = synth
    m = HierTypedMF(rank=3, lam_u=0.03, lam_v=0.01, reconcile=False).fit(md)
    assert np.degrees(subspace_angles(d.V, m.V_)).max() < 5
    assert np.degrees(subspace_angles(d.U - d.U.mean(0), m.U_ - m.U_.mean(0))).max() < 15


def test_hierarchy_helps_and_couplings_run(synth):
    _, md, Z = synth
    test = ~md.mask
    err = {
        c: rmse(HierTypedMF(rank=3, lam_u=0.3, coupling=c, reconcile=False).fit(md).predict(), Z, test)
        for c in ("none", "broadcast", "prior_only", "prior_and_constraint")
    }
    assert err["prior_only"] < err["none"]
    assert err["prior_and_constraint"] <= err["prior_only"] + 0.01
    with pytest.raises(ValueError):
        HierTypedMF(coupling="nope")


def test_reconciliation_is_coherent_and_helps(synth):
    _, md, Z = synth
    m = HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md)
    P = m.predict()
    R = reconcile(md, P, m.predict_std() ** 2)
    assert rmse(R, Z, ~md.mask) <= rmse(P, Z, ~md.mask)
    # Synthetic values are centred near zero, so the gap is measured against the column's spread.
    gaps = np.abs(np.median(coherence_error(md, R)))
    assert gaps < 0.01
    assert np.array_equal(R[md.mask], P[md.mask])  # observed cells untouched


def test_level_transfer_uses_aggregation(synth):
    _, md, Z = synth
    level = md.level
    test = np.zeros_like(md.mask)
    test[level == 2, 0] = md.mask[level == 2, 0]
    tr = md.with_mask(md.mask & ~test)
    e_prior = rmse(
        HierTypedMF(rank=3, lam_u=0.3, coupling="prior_only", reconcile=False).fit(tr).predict(), Z, test
    )
    e_agg = rmse(HierTypedMF(rank=3, lam_u=0.3, lam_agg=10).fit(tr).predict(), Z, test)
    assert e_agg < e_prior


def test_laplace_intervals_roughly_calibrated(synth):
    _, md, Z = synth
    rng = np.random.default_rng(0)
    test = md.mask & (rng.random(md.mask.shape) < 0.15)
    m = HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md.with_mask(md.mask & ~test))
    sd = m.predict_std()
    cover = np.mean(np.abs(m.predict() - Z)[test] <= 1.645 * sd[test])
    assert 0.75 < cover < 0.98
    S = m.predict_dist(n_samples=50, seed=1)
    assert S.shape == (50, *Z.shape)
    assert np.allclose(S.mean(axis=0)[test], m.predict_latent()[test], atol=0.15)


def test_fold_in_close_to_refit(synth):
    _, md, Z = synth
    rng = np.random.default_rng(3)
    rows = rng.choice(np.flatnonzero(md.level == 2), 15, replace=False)
    hide = np.zeros_like(md.mask)
    for g in rows:
        obs = np.flatnonzero(md.mask[g])
        hide[g, rng.choice(obs, len(obs) // 2, replace=False)] = True
    full = HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md.with_mask(md.mask & ~hide))
    without = md.mask.copy()
    without[rows] = False
    base = HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md.with_mask(without))
    keep = md.mask[rows] & ~hide[rows]
    F = base.fold_in(np.where(keep, md.Z[rows], 0.0), keep, parent_new=[md.parent[g] for g in rows])
    e_full = rmse(full.predict()[rows], Z[rows], hide[rows])
    e_fold = rmse(F, Z[rows], hide[rows])
    assert abs(e_fold - e_full) < 0.05
    assert base.fold_in_cov_.shape == (15, 4, 4)


def test_fold_in_indicator(synth):
    _, md, Z = synth
    m = HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(
        replace(
            md, Z=md.Z[:, 1:], mask=md.mask[:, 1:], column_meta=md.column_meta[1:], scalers=md.scalers[1:]
        )
    )
    obs = md.mask[:, 0]
    pred = m.fold_in_indicator(md.Z[:, 0], obs)
    assert rmse(pred[:, None], Z[:, [0]], ~obs[:, None]) < 0.6 * np.std(Z[:, 0])


def typed_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, 2))
    V = rng.standard_normal((6, 2))
    lat = U @ V.T
    Z = lat.copy()
    Z[:, 4] = np.digitize(lat[:, 4], [-1.0, 0.0, 1.0])  # ordinal 0..3
    cat = np.stack([lat[:, 5], -lat[:, 5], 0.5 * lat[:, 0]], axis=1)
    Z[:, 5] = cat.argmax(axis=1)  # categorical 0..2
    mask = rng.random(Z.shape) > 0.2
    meta = [Indicator(f"c{j}", DType.CONTINUOUS) for j in range(4)]
    meta += [
        Indicator("o", DType.ORDINAL, categories=("0", "1", "2", "3")),
        Indicator("k", DType.CATEGORICAL, categories=("a", "b", "c")),
    ]
    data = ModelData(
        Z=np.where(mask, Z, np.nan),
        mask=mask,
        parent=[[] for _ in range(n)],
        agg=None,
        column_meta=meta,
        level=np.zeros(n, int),
    )
    return data, Z


def test_typed_losses():
    data, Z = typed_data()
    m = HierTypedMF(rank=2, lam_u=0.01, lam_v=0.01, coupling="none", reconcile=False).fit(data)
    P = m.predict()
    test = ~data.mask
    ordinal_acc = np.mean(np.rint(P[test[:, 4], 4]) == Z[test[:, 4], 4])
    cat_acc = np.mean(P[test[:, 5], 5] == Z[test[:, 5], 5])
    assert ordinal_acc > 0.6 and cat_acc > 0.6
    pr = m.predict_proba(4)
    assert pr.shape == (Z.shape[0], 4) and np.allclose(pr.sum(axis=1), 1)
    assert np.all(np.diff(m.thresholds_[4]) > 0)
    S = m.predict_dist(n_samples=5)
    assert set(np.unique(S[:, :, 5])) <= {0.0, 1.0, 2.0}


def test_huber_and_side_information(synth):
    _, md, Z = synth
    side = np.random.default_rng(0).standard_normal((md.Z.shape[0], 2))
    m = HierTypedMF(rank=3, lam_u=0.3, loss="huber", reconcile=False).fit(replace(md, side=side))
    assert m.A_.shape == (2, 3) and np.isfinite(m.predict()).all()


def test_spatial_and_time_penalties(synth):
    from scipy import sparse

    _, md, _ = synth
    n = md.Z.shape[0]
    adj = sparse.random(n, n, density=0.01, random_state=0, format="csr")
    adj = ((adj + adj.T) > 0).astype(float)
    pairs = np.array([[i, i + 1] for i in range(0, n - 1, 2)])
    m = HierTypedMF(rank=3, lam_lap=1.0, lam_time=5.0, reconcile=False).fit(
        replace(md, adjacency=adj, time_pairs=pairs)
    )
    gap = np.linalg.norm(m.U_[pairs[:, 0]] - m.U_[pairs[:, 1]], axis=1).mean()
    m0 = HierTypedMF(rank=3, reconcile=False).fit(md)
    gap0 = np.linalg.norm(m0.U_[pairs[:, 0]] - m0.U_[pairs[:, 1]], axis=1).mean()
    assert gap < gap0


def test_broadcast_fill():
    Z = np.array([[1.0, 2.0], [np.nan, 5.0], [np.nan, np.nan]])
    d = ModelData(
        Z=Z,
        mask=np.isfinite(Z),
        parent=[[], [(0, 1.0)], [(1, 1.0)]],
        agg=None,
        column_meta=[],
        level=np.array([0, 1, 2]),
    )
    F, M = broadcast_fill(d)
    assert M.all() and F[2, 0] == 1.0 and F[2, 1] == 5.0


def test_reproducible(synth):
    _, md, _ = synth
    a = HierTypedMF(rank=3, seed=5).fit(md).predict()
    b = HierTypedMF(rank=3, seed=5).fit(md).predict()
    assert np.array_equal(a, b)
