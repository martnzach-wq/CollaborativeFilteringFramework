import numpy as np
import pytest

from regsim.evaluation import masking
from regsim.evaluation.benchmark import model_card, prototype_splits, run_benchmark, summarise
from regsim.evaluation.metrics import coverage, crps_gaussian, crps_samples, mae, r2_per_column, rmse
from regsim.models.base import ModelData
from regsim.models.baselines import LevelMean, ParentCarryDown
from regsim.schema import DType, Indicator


def grid(n=60, p=6, seed=0):
    rng = np.random.default_rng(seed)
    level = np.repeat([0, 2, 3], [4, 16, 40])
    country = np.array([f"C{i % 4}" for i in range(n)])
    Z = rng.standard_normal((n, p))
    mask = rng.random((n, p)) > 0.2
    mask[level == 0] = True
    meta = [Indicator(f"i{j}", DType.CONTINUOUS, native_levels=(str([2, 3][j % 2]),)) for j in range(p)]
    parent = [[] if lv == 0 else [(i % 4, 1.0)] for i, lv in enumerate(level)]
    return ModelData(
        Z=np.where(mask, Z, np.nan),
        mask=mask,
        parent=parent,
        agg=None,
        column_meta=meta,
        level=level,
        country=country,
        column_ids=[m.indicator_id for m in meta],
    )


def test_random_cells():
    d = grid()
    s = masking.random_cells(d.mask, 0.1, seed=1)
    assert s.test.sum() == int(0.1 * d.mask.sum()) and not (s.test & ~d.mask).any()
    assert np.array_equal(s.test, masking.random_cells(d.mask, 0.1, seed=1).test)


def test_cold_start_keeps_k():
    d = grid()
    splits = masking.entity_cold_start(d.mask, d.level, 2, frac=0.5, k_keep=(1, 3), min_observed=4, seed=2)
    assert [s.k for s in splits] == [1, 3]
    for s in splits:
        rows = np.flatnonzero(s.test.any(axis=1))
        assert (d.level[rows] == 2).all()
        kept = (d.mask & ~s.test)[rows].sum(axis=1)
        assert (kept == s.k).all()


def test_country_blocks_and_level_transfer():
    d = grid()
    s = masking.country_blocks(
        d.mask, d.level, d.country, [0, 2], target_level=2, n_blocks=3, min_cells=2, seed=0
    )
    cols = np.flatnonzero(s.test.any(axis=0))
    assert set(cols) <= {0, 2}
    for j in cols:
        assert len(set(d.country[s.test[:, j]])) >= 1 and (d.level[s.test[:, j]] == 2).all()
    lt = masking.level_transfer(d.mask, d.level, 1, 3, "i1")
    assert np.array_equal(lt.test[:, 1], d.mask[:, 1] & (d.level == 3)) and lt.test[:, 0].sum() == 0


def test_metrics():
    t = np.array([[0.0, 1.0], [2.0, 3.0]])
    p = t + np.array([[1.0, -1.0], [0.0, 0.0]])
    m = np.ones_like(t, dtype=bool)
    assert rmse(p, t, m) == pytest.approx(np.sqrt(0.5)) and mae(p, t, m) == pytest.approx(0.5)
    assert coverage(p - 0.5, p + 0.5, t, m) == pytest.approx(0.5)
    r2 = r2_per_column(t, t, m)
    assert np.allclose(r2, 1.0)
    # CRPS of a sharp correct forecast is about zero; sample-based CRPS approximates the Gaussian one.
    assert crps_gaussian(t, np.full_like(t, 1e-9), t, m) == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    samples = t[None] + rng.standard_normal((4000, 2, 2))
    assert crps_samples(samples, t, m) == pytest.approx(crps_gaussian(t, np.ones_like(t), t, m), rel=0.05)


def test_runner_and_model_card(tmp_path):
    d = grid()
    splits = prototype_splits(d, m1_seeds=(0,), m2_seeds=(1,), m3_seeds=(2,))
    schemes = {s.scheme[:2] for s in splits}
    assert schemes == {"M1", "M2", "M3", "M4"}
    res = run_benchmark(d, {"level": LevelMean, "carry": ParentCarryDown}, splits)
    assert len(res) == 2 * sum(s.test.any() for s in splits)  # empty splits are skipped
    assert {"rmse", "mae", "crps", "coverage90", "width90"} <= set(res.columns)
    s = summarise(res)
    assert set(s["rank"]) <= {1, 2}
    md = model_card(res, tmp_path, {"data": "synthetic"}).read_text()
    assert "M1 random cells" in md and "M4 level transfer" in md
    assert (tmp_path / "benchmark.csv").exists() and (tmp_path / "summary.csv").exists()


def test_estimator_factories_and_config():
    from regsim.config import RunConfig
    from regsim.evaluation.run import estimator_factories

    cfg = RunConfig.from_yaml("configs/benchmark_prototype.yaml")
    f = estimator_factories(cfg.evaluation.estimators)
    assert f["KNNGower"]().k == 10 and f["SoftImpute"]().lam == 5.0
    with pytest.raises(ValueError, match="unknown estimator"):
        estimator_factories({"Nope": {}})
