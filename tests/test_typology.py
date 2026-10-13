import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from regsim.schema import Lens
from regsim.synthetic import generate_hierarchical
from regsim.typology.run import TypologyRun, run_typology, track

warnings.filterwarnings("ignore")


class Planted:
    """Stand-in model whose completed profiles are four well-separated clusters."""

    def __init__(self, n, p, seed=0):
        rng = np.random.default_rng(seed)
        self.truth = rng.integers(0, 4, n)
        centres = rng.normal(0, 3, (4, p))
        self.P = centres[self.truth] + rng.normal(0, 0.5, (n, p))

    def predict(self):
        return self.P

    def get_params(self):
        return {"planted": 4}


@pytest.fixture(scope="module")
def planted():
    md = generate_hierarchical(n_per_level=(3, 20, 120), n_indicators=6, rank=2, missingness="none", seed=1)
    md = md.model_data()
    return md, Planted(*md.Z.shape)


@pytest.mark.parametrize("algorithm", ["gmm", "ward"])
def test_recovers_planted_clusters_and_k(planted, algorithm):
    md, model = planted
    ty = run_typology(model, md, TypologyRun(algorithm=algorithm, k_range=(2, 6), n_boot=8))
    assert ty.k == 4
    assert adjusted_rand_score(model.truth, ty.labels) > 0.95
    assert ty.stability > 0.9 and np.nanmin(ty.entity_stability) > 0.5
    assert set(ty.cluster_table.columns) >= {"size", "label", "dispersion", "stability"}
    assert ty.cluster_table["size"].sum() == md.Z.shape[0]
    assert ty.cluster_table.label.str.match(r"(high|low) ind_").all()


def test_hdbscan_and_lens_levels(planted):
    md, model = planted
    ty = run_typology(model, md, TypologyRun(lens=Lens(levels=(2,)), algorithm="hdbscan", min_cluster_size=5))
    assert len(ty.labels) == int((md.levels() == 2).sum())
    assert ty.k >= 3


def test_reproducible_hash_and_tracking(planted):
    md, model = planted
    run = TypologyRun(algorithm="gmm", k=4, n_boot=4, seed=3)
    a, b = run_typology(model, md, run), run_typology(model, md, run)
    assert a.config_hash == b.config_hash and np.array_equal(a.labels, b.labels)
    assert a.config_hash != TypologyRun(algorithm="gmm", k=4, n_boot=4, seed=4).config_hash(model, md)
    c = run_typology(model, md, TypologyRun(algorithm="ward", k=4, n_boot=4))
    mapping, trans = track(a, c)
    assert sorted(mapping) == [0, 1, 2, 3]
    assert np.trace(trans.to_numpy()) == trans.to_numpy().sum()  # same clusters after alignment


def test_spatial_ward_needs_adjacency(planted):
    md, model = planted
    with pytest.raises(ValueError):
        run_typology(model, md, TypologyRun(algorithm="ward_spatial", k=4, n_boot=0))
    n = md.Z.shape[0]
    md.adjacency = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
    ty = run_typology(model, md, TypologyRun(algorithm="ward_spatial", k=4, n_boot=0))
    assert ty.k == 4
    md.adjacency = None
