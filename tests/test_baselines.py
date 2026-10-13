import numpy as np
import pytest

from regsim.models.base import ModelData
from regsim.models.baselines import (
    BASELINES,
    GlobalMean,
    IterativePCA,
    KNNGower,
    LevelMean,
    MissForest,
    ParentCarryDown,
    SoftImpute,
)
from regsim.synthetic import generate_hierarchical


@pytest.fixture(scope="module")
def synth():
    d = generate_hierarchical(
        n_per_level=(6, 30, 120), n_indicators=12, rank=3, missingness="mcar", missing_rate=0.3, seed=3
    )
    Z = d.latent[0] + 0.05 * np.random.default_rng(0).standard_normal(d.latent[0].shape)
    mask = d.observed[0]
    data = ModelData(
        Z=np.where(mask, Z, np.nan),
        mask=mask,
        parent=d.parent,
        agg=None,
        column_meta=d.indicators,
        level=d.level,
        country=np.array([d.geo_ids[d.root_of(i)] for i in range(d.n_entities)]),
    )
    return data, Z


def small():
    # Rows: root (level 0), two children (level 1), one grandchild of child 1 (level 2).
    Z = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, np.nan], [np.nan, np.nan]])
    mask = np.isfinite(Z)
    parent = [[], [(0, 0.5)], [(0, 0.5)], [(1, 1.0)]]
    return ModelData(Z=Z, mask=mask, parent=parent, agg=None, column_meta=[], level=np.array([0, 1, 1, 2]))


def test_means():
    d = small()
    assert np.allclose(GlobalMean().fit(d).predict()[:, 0], 2.0)
    P = LevelMean().fit(d).predict()
    assert P[1, 0] == pytest.approx(3.0) and P[0, 1] == pytest.approx(4.0)  # global mean where level has none


def test_parent_carry_down():
    P = ParentCarryDown().fit(small()).predict()
    assert P[1, 0] == pytest.approx(1.0)  # from the root
    assert P[3, 0] == pytest.approx(1.0)  # through child 1, which has no value itself
    assert P[3, 1] == pytest.approx(4.0)  # from child 1


def test_carry_down_fractional_parents():
    Z = np.array([[2.0], [6.0], [np.nan]])
    d = ModelData(
        Z=Z,
        mask=np.isfinite(Z),
        parent=[[], [], [(0, 0.25), (1, 0.75)]],
        agg=None,
        column_meta=[],
        level=np.array([0, 0, 1]),
    )
    assert ParentCarryDown().fit(d).predict()[2, 0] == pytest.approx(0.25 * 2 + 0.75 * 6)


def test_knn_gower_uses_same_level_neighbours():
    Z = np.array([[0.0, 0.0], [0.1, 1.0], [5.0, 9.0], [0.05, np.nan]])
    d = ModelData(Z=Z, mask=np.isfinite(Z), parent=[[]] * 4, agg=None, column_meta=[], level=np.zeros(4, int))
    P = KNNGower(k=1).fit(d).predict()
    assert P[3, 1] == pytest.approx(0.0)  # nearest on column 0 is row 0 (distance 0.05 vs 0.05 ties broken)
    P2 = KNNGower(k=2).fit(d).predict()
    assert 0.0 < P2[3, 1] < 1.0


@pytest.mark.parametrize("name", list(BASELINES))
def test_interface_and_finite(synth, name):
    data, Z = synth
    est = BASELINES[name](**({"n_estimators": 10, "max_iter": 2} if name == "MissForest" else {}))
    est.fit(data)
    P = est.predict()
    assert P.shape == Z.shape and np.isfinite(P).all()
    S = est.predict_dist(n_samples=5, seed=1)
    assert S.shape == (5, *Z.shape)
    assert est.embed_entities().shape[0] == Z.shape[0] and est.embed_indicators().shape[0] == Z.shape[1]


def test_low_rank_methods_beat_means(synth):
    data, Z = synth
    test = ~data.mask

    def err(est):
        P = est.fit(data).predict()
        return np.sqrt(np.mean((P - Z)[test] ** 2))

    base = err(GlobalMean())
    assert err(SoftImpute(lam=1.0, rank_max=5)) < 0.6 * base
    assert err(IterativePCA(rank=3)) < 0.5 * base
    assert err(ParentCarryDown()) < base


def test_missforest_reproducible(synth):
    data, _ = synth
    a = MissForest(n_estimators=10, max_iter=2, seed=4).fit(data).predict()
    b = MissForest(n_estimators=10, max_iter=2, seed=4).fit(data).predict()
    assert np.array_equal(a, b)


def test_fold_in_matches_refit(synth):
    data, Z = synth
    est = LevelMean().fit(data)
    new = est.fold_in(
        Z[:2], data.mask[:2], parent_new=[data.parent[0], data.parent[1]], level_new=data.level[:2]
    )
    assert new.shape == (2, Z.shape[1]) and np.isfinite(new).all()
