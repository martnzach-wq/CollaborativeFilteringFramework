import warnings

import numpy as np
import pytest

from regsim.evaluation import masking
from regsim.models.baselines import LevelMean
from regsim.models.hier_mf import HierTypedMF
from regsim.synthetic import generate_hierarchical
from regsim.uncertainty.conformal import (
    ConformalCalibrator,
    Conformalised,
    calibration_split,
    mondrian_groups,
)
from regsim.uncertainty.domain import applicability

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def synthetic():
    d = generate_hierarchical(
        n_per_level=(6, 40, 200),
        n_indicators=12,
        rank=3,
        noise=0.2,
        missingness="mcar",
        missing_rate=0.2,
        seed=5,
    )
    return d.model_data()


def test_conformal_quantile_and_fallbacks():
    scores = np.arange(1, 100, dtype=float)
    groups = np.r_[np.full(80, 21), np.full(19, 32)]
    cal = ConformalCalibrator.fit(scores, groups, alpha=0.1, min_count=50)
    assert cal.overall == 90.0  # ceil(100 * 0.9) = 90th of 99 sorted scores
    assert cal.quantiles[21] == 73.0  # ceil(81 * 0.9) = 73
    assert 32 not in cal.quantiles  # too few cells: falls back
    assert cal.quantile(np.array([21, 32, 41])).tolist() == [73.0, 90.0, 73.0]


def test_mondrian_groups_and_calibration_split(synthetic):
    d = synthetic
    level = d.levels()
    g = mondrian_groups(level, d.mask)
    assert set(np.unique(g // 10)) == set(np.unique(level))
    cal = calibration_split(d.mask, level, seed=0)
    assert not (cal & ~d.mask).any()
    cold = (d.mask & ~cal).sum(axis=1) <= 10
    assert cold.any() and 0.05 < cal.sum() / d.mask.sum() < 0.4


@pytest.mark.parametrize("factory", [lambda: HierTypedMF(rank=3, lam_u=0.3, reconcile=False), LevelMean])
def test_conformal_coverage_on_synthetic(synthetic, factory):
    d = synthetic
    covs = []
    for seed in range(2):
        split = masking.random_cells(d.mask, 0.15, seed)
        m = Conformalised(factory, seed=seed).fit(d.with_mask(split.train(d.mask)))
        lo, hi = m.predict_interval()
        covs.append(np.mean((np.abs(d.Z - (lo + hi) / 2) <= (hi - lo) / 2)[split.test]))
        assert np.all(m.predict_std()[np.isfinite(m.predict_std())] > 0)
    assert 0.85 <= np.mean(covs) <= 0.95


def test_applicability_flags(synthetic):
    d = synthetic
    split = masking.entity_cold_start(d.mask, d.levels(), 2, frac=0.2, k_keep=(1,), min_observed=5, seed=0)[0]
    tr = d.with_mask(split.train(d.mask))
    ad = applicability(HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(tr), tr)
    cold = split.test.any(axis=1)
    assert set(ad.domain) <= {"in_domain", "edge", "outside"}
    assert (ad.domain[cold] == "outside").all()  # one observed cell
    assert (ad.domain[~cold] == "in_domain").mean() > 0.85
