import warnings

import numpy as np
import pytest

from regsim.evaluation import masking
from regsim.evaluation.neighbours import neighbour_skill
from regsim.models.hier_mf import HierTypedMF
from regsim.schema import Lens
from regsim.similarity.entities import (
    ProfileSpace,
    nearest,
    percentile_score,
    resolve_lens,
    s_gower,
    s_latent,
    s_profile,
)
from regsim.similarity.indicators import i_model, i_observed, i_partial, i_predictive
from regsim.similarity.transfer import learn_transfer_weights
from regsim.synthetic import generate_hierarchical

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fitted():
    d = generate_hierarchical(
        n_per_level=(6, 30, 150),
        n_indicators=10,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        seed=4,
    )
    md = d.model_data()
    return d, md, HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md)


def test_lens_resolution(fitted):
    _, md, _ = fitted
    li = resolve_lens(md, Lens(indicators=tuple(md.column_ids[:3]), levels=(2,)))
    assert list(li.cols) == [0, 1, 2] and (md.level[li.rows] == 2).all()
    with pytest.raises(KeyError):
        resolve_lens(md, Lens(indicators=("nope",)))


def test_profile_space_whitening_and_contributions():
    rng = np.random.default_rng(0)
    from regsim.models.base import ModelData

    X = rng.standard_normal((200, 3)) @ np.array([[1, 0.9, 0], [0, 0.3, 0], [0, 0, 2.0]])
    md = ModelData(
        Z=X,
        mask=np.ones_like(X, bool),
        parent=[[]] * 200,
        agg=None,
        column_meta=[],
        level=np.zeros(200, int),
        column_ids=["a", "b", "c"],
        geo_ids=[f"g{i}" for i in range(200)],
    )
    sp = ProfileSpace(X, md, ridge=0.0)
    assert np.allclose(np.cov(sp.Y, rowvar=False), np.eye(3), atol=1e-8)
    c = sp.contributions(0, 1)
    assert c.sum() == pytest.approx(1.0)
    assert sp.distances(0, np.array([1]))[0] == pytest.approx(np.sqrt(((sp.Y[0] - sp.Y[1]) ** 2).sum()))
    ref = sp.pair_distribution(0)
    assert percentile_score(np.array([0.0]), ref)[0] == pytest.approx(100.0)
    assert percentile_score(np.array([ref.max()]), ref)[0] == pytest.approx(0.0)


def test_s_profile_finds_hierarchical_siblings(fitted):
    d, md, m = fitted
    sp = s_profile(m, md, Lens(levels=(2,)))
    g = int(np.flatnonzero(md.level == 2)[0])
    nn = nearest(sp, md.geo_ids[g], k=5, samples=m.predict_dist(20, noise=False))
    assert len(nn) == 5 and (nn.level == 2).all()
    assert (nn.distance_lo <= nn.distance).all() or (nn.distance <= nn.distance_hi).all()
    # Neighbours are closer in the true factor space than random entities of the level.
    idx = [md.geo_ids.index(x) for x in nn.geo_id]
    rows = np.flatnonzero(md.level == 2)
    true_nn = np.linalg.norm(d.U[idx] - d.U[g], axis=1).mean()
    true_all = np.linalg.norm(d.U[rows] - d.U[g], axis=1).mean()
    assert true_nn < 0.6 * true_all
    assert s_latent(m, md).Y.shape[0] == md.Z.shape[0]


def test_s_gower(fitted):
    _, md, _ = fitted
    dist, cov = s_gower(md, 0, np.arange(1, 6))
    assert dist.shape == (5,) and ((cov > 0) & (cov <= 1)).all()
    assert np.nanmax(dist) <= 1.0


def test_indicator_measures(fitted):
    _, md, m = fitted
    im = i_model(m, md)
    assert np.allclose(np.diag(im), 1.0) and np.allclose(im, im.T)
    io = i_observed(md, min_pairs=10)
    assert len(io) == 45 and (io.n > 0).all() and ((io.lo <= io.rho) & (io.rho <= io.hi)).all()
    ip = i_partial(m, md, alpha=0.01)
    assert ip.shape == (10, 10)
    x = [im.loc[r.indicator_a, r.indicator_b] for r in io.itertuples()]
    assert np.corrcoef(x, io.rho)[0, 1] > 0.8
    ipred = i_predictive(
        lambda: HierTypedMF(rank=3, lam_u=0.3, reconcile=False, max_iter=100),
        md,
        md.column_ids[0],
        md.column_ids[1:3],
    )
    assert set(ipred.index) == set(md.column_ids[1:3])


def test_transfer_weights_find_relevant_indicator():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 4))
    y = 2 * X[:, 2] + 0.1 * rng.standard_normal(60)
    tw = learn_transfer_weights(X, np.arange(60), y, ["a", "b", "c", "d"], lam=0.01, n_boot=10)
    assert tw.weights.idxmax() == "c"
    assert tw.loo_rmse < tw.loo_rmse_default
    assert (tw.lower <= tw.upper).all()


def test_neighbour_skill_profile_beats_gower(fitted):
    _, md, _ = fitted
    splits = masking.entity_cold_start(md.mask, md.level, 2, frac=0.3, k_keep=(2,), min_observed=5, seed=0)
    r = neighbour_skill(md, splits, lambda: HierTypedMF(rank=3, lam_u=0.3, reconcile=False)).set_index(
        "measure"
    )
    assert r.loc["S-profile", "rmse"] < r.loc["S-gower", "rmse"]
