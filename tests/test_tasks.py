import warnings

import numpy as np
import pytest

from regsim.models.hier_mf import HierTypedMF
from regsim.predict import tasks as T
from regsim.prep.matrix import to_original
from regsim.synthetic import generate_hierarchical

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fitted():
    d = generate_hierarchical(
        n_per_level=(4, 20, 80),
        n_indicators=8,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        seed=1,
    ).model_data()
    return d, HierTypedMF(rank=3, lam_u=0.3).fit(d)


def test_fill_gaps_marks_observed(fitted):
    d, m = fitted
    out = T.fill_gaps(m)
    assert len(out) == d.Z.size
    orig = to_original(d, np.where(d.mask, d.Z, 0.0))
    obs = out[out.observed]
    rows = [d.geo_ids.index(g) for g in obs.geo_id]
    cols = [d.column_ids.index(c) for c in obs.indicator]
    assert np.allclose(obs.value, orig[rows, cols])
    pred = out[~out.observed]
    assert (pred.lower < pred.value).all() and (pred.value < pred.upper).all()


def test_new_entity_and_vision(fitted):
    d, m = fitted
    g = d.geo_ids[30]
    present = T.fill_gaps(m, g).set_index("indicator").value
    c0 = d.column_ids[0]
    new = T.new_entity(m, {c0: present[c0]}, parent=d.geo_ids[d.parent[30][0][0]]).set_index("indicator")
    assert new.loc[c0, "observed"] and np.isclose(new.loc[c0, "value"], present[c0])
    assert new.drop(index=c0).upper.gt(new.drop(index=c0).lower).all()
    vis = T.vision_entity(m, g, {c0: present[c0] + 1.0}).set_index("indicator")
    assert np.isclose(vis.loc[c0, "vision"], present[c0] + 1.0) and vis.loc[c0, "target"]
    assert vis.drop(index=c0).change.abs().max() > 0


def test_analogues_and_next_indicator(fitted):
    d, m = fitted
    g = d.geo_ids[30]
    an = T.analogues(m, g, k=5)
    assert len(an) == 5 and g not in set(an.geo_id) and an.distance.is_monotonic_increasing
    nxt = T.next_indicator(m, g, targets=[d.column_ids[-1]], candidates=d.column_ids[:-1])
    assert (nxt.variance_reduction >= 0).all() and nxt.variance_reduction.is_monotonic_decreasing
    assert nxt.share.sum() < 1.0 + 1e-9


def test_new_indicator_and_level_transfer(fitted):
    d, m = fitted
    rng = np.random.default_rng(0)
    truth = d.Z[:, 2]
    known = {d.geo_ids[i]: float(truth[i]) for i in rng.choice(len(truth), 60, replace=False)}
    col = T.new_indicator(m, known)
    assert len(col) == len(d.geo_ids)
    assert np.corrcoef(col.to_numpy()[np.isfinite(truth)], truth[np.isfinite(truth)])[0, 1] > 0.8
    lt = T.level_transfer(m, d.column_ids[0], 2)
    assert len(lt) == int((d.levels() == 2).sum()) and lt.parent.str.len().gt(0).all()
