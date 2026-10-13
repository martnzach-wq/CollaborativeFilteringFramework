import numpy as np
import pytest

from regsim.schema import Aggregation, DType
from regsim.synthetic import generate_hierarchical

MIXED = {"continuous": 0.4, "share": 0.2, "count": 0.2, "ordinal": 0.2}
SMALL = {"n_per_level": (5, 20, 80), "n_indicators": 12, "rank": 3}


def _arrays(d):
    return [d.U, d.tau, d.V, d.mu, d.level, d.population, d.area, d.latent, d.X, d.observed]


@pytest.mark.parametrize("missingness", ["none", "mcar", "block_mnar", "level"])
def test_reproducible_by_seed(missingness):
    a = generate_hierarchical(**SMALL, dtypes=MIXED, missingness=missingness, n_periods=2, seed=7)
    b = generate_hierarchical(**SMALL, dtypes=MIXED, missingness=missingness, n_periods=2, seed=7)
    c = generate_hierarchical(**SMALL, dtypes=MIXED, missingness=missingness, n_periods=2, seed=8)
    for x, y in zip(_arrays(a), _arrays(b), strict=True):
        assert np.array_equal(x, y)
    assert a.parent == b.parent and a.indicators == b.indicators
    assert not np.array_equal(a.U, c.U)


def test_shapes_and_hierarchy():
    d = generate_hierarchical(**SMALL, n_periods=2, seed=0)
    n = sum(SMALL["n_per_level"])
    assert d.U.shape == (n, 3) and d.V.shape == (12, 3)
    assert d.X.shape == d.observed.shape == (2, n, 12)
    assert np.bincount(d.level).tolist() == list(SMALL["n_per_level"])
    for c, parents in enumerate(d.parent):
        assert (len(parents) == 0) == (d.level[c] == 0)
        assert all(d.level[p] == d.level[c] - 1 for p, _ in parents)
    # Every parent has at least one child.
    has_child = {p for parents in d.parent for p, _ in parents}
    assert has_child == set(np.flatnonzero(d.level < 2))
    assert len(set(d.geo_ids)) == n


def test_population_shares_sum_to_one():
    d = generate_hierarchical(**SMALL, fractional_share=0.3, seed=1)
    A = d.aggregation_matrix("pop")
    sums = np.asarray(A.sum(axis=1)).ravel()
    assert np.allclose(sums[d.level < 2], 1.0)
    assert any(len(p) == 2 for p in d.parent)
    M = d.aggregation_matrix("membership")
    assert np.allclose(M @ d.population, np.where(d.level < 2, d.population, 0.0))


def test_mixed_types_on_original_scale():
    d = generate_hierarchical(**SMALL, dtypes=MIXED, missingness="none", seed=2)
    counts = {t: sum(ind.dtype is t for ind in d.indicators) for t in DType}
    assert sum(counts.values()) == 12 and counts[DType.CONTINUOUS] >= 4
    for k, ind in enumerate(d.indicators):
        x = d.X[..., k]
        if ind.dtype is DType.SHARE:
            assert np.all((x > 0) & (x < 1))
        elif ind.dtype is DType.COUNT:
            assert np.all(x >= 0)
        elif ind.dtype is DType.ORDINAL:
            assert set(np.unique(x)) <= set(range(5))


@pytest.mark.parametrize("fractional_share", [0.0, 0.3])
def test_coarse_values_are_exact_aggregates(fractional_share):
    d = generate_hierarchical(
        **SMALL, dtypes=MIXED, missingness="none", fractional_share=fractional_share, seed=3
    )
    W = d.aggregation_matrix("pop").toarray()
    M = d.aggregation_matrix("membership").toarray()
    coarse = d.level < 2
    for k, ind in enumerate(d.indicators):
        x = d.X[0, :, k]
        if ind.aggregation is Aggregation.WMEAN_POP:
            assert np.allclose((W @ x)[coarse], x[coarse])
        elif ind.aggregation is Aggregation.SUM:
            assert np.allclose((M @ x)[coarse], x[coarse])


def test_missingness_patterns():
    kw = {**SMALL, "n_indicators": 40}
    full = generate_hierarchical(**kw, missingness="none", seed=4)
    assert full.observed.all()
    mcar = generate_hierarchical(**kw, missingness="mcar", missing_rate=0.3, seed=4)
    assert abs(1 - mcar.observed.mean() - 0.3) < 0.03
    mnar = generate_hierarchical(**kw, missingness="block_mnar", missing_rate=0.3, seed=4)
    assert 0.15 < 1 - mnar.observed.mean() < 0.45
    z = mnar.latent[0]
    zs = (z - z.mean(axis=0)) / z.std(axis=0)
    # High values are missing more often than low values.
    assert (~mnar.observed[0])[zs > 1].mean() > (~mnar.observed[0])[zs < -1].mean()
    lvl = generate_hierarchical(**kw, missingness="level", seed=4)
    share_by_level = [lvl.observed[0, lvl.level == lv].mean() for lv in range(3)]
    assert share_by_level[0] > share_by_level[1] > share_by_level[2]
    # Indicators observed only at level 0 exist, as in Eurostat data.
    assert any(ind.native_levels == ("0",) for ind in lvl.indicators)


def test_random_walk_over_periods():
    d = generate_hierarchical(**SMALL, n_periods=3, drift=0.2, seed=5)
    assert np.all(d.tau[0] == 0)
    step = d.tau[2] - d.tau[1]
    assert 0.15 < step.std() < 0.25
    assert d.periods == ["P0", "P1", "P2"]


def test_low_rank_latent():
    d = generate_hierarchical(**SMALL, missingness="none", seed=6)
    centred = d.latent[0] - d.mu
    s = np.linalg.svd(centred, compute_uv=False)
    assert s[3] < 1e-8 * s[0]


def test_observations_long_format():
    d = generate_hierarchical(**SMALL, missingness="mcar", n_periods=2, seed=9)
    obs = d.to_observations()
    assert len(obs) == d.observed.sum()
    assert {"geo_id", "indicator_id", "period", "value", "source", "source_version", "flags"} <= set(
        obs.columns
    )
    assert obs["geo_id"].str.startswith("SYN:").all()
    assert len(d.units()) == d.n_entities
    assert len(d.edges()) == sum(len(p) for p in d.parent)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_hierarchical(n_per_level=(10, 5), sigma_level=(1.0, 0.5))
    with pytest.raises(ValueError):
        generate_hierarchical(**SMALL, missingness="unknown")
    with pytest.raises(ValueError):
        generate_hierarchical(**SMALL, dtypes={"categorical": 1.0})
