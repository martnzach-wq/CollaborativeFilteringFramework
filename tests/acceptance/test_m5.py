"""M5 acceptance (HANDOVER.md §3).

1. Empirical coverage of nominal 90 % intervals between 0.87 and 0.93 on the M1 and M2 hold-outs (prototype
   protocol, Eurostat 2018/2022) at each level, with split-conformal calibration.
2. Identical typology for an identical configuration hash (model fitted and typology run twice).
3. All D3.6 gates G001-G010 expressible in the rule file; with the GeoPackage, the published gate flags and
   eligibility of all demonstrators are reproduced.

Tests 1 and 2 need a completed `regsim build` (about 3 minutes on 4 cores).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regsim.evaluation.benchmark import Z90, prototype_splits
from regsim.models.hier_mf import HierTypedMF
from regsim.prep.matrix import prepare
from regsim.registry import IndicatorRegistry
from regsim.rules.engine import evaluate, gate_summary, load_rules
from regsim.schema import Lens
from regsim.similarity.d36 import load_values, published_gates
from regsim.store.store import ObservationStore
from regsim.typology.run import TypologyRun, run_typology
from regsim.uncertainty.conformal import Conformalised

warnings.filterwarnings("ignore")
STORE = Path("data/store")
GPKG = Path("data/raw/d36/knowing-transferability-export-v2.5.gpkg")
needs_build = pytest.mark.skipif(not (STORE / "observations").exists(), reason="run `regsim build` first")
DEMONSTRATORS = {
    "EE@LAU:4:0784": "urban",
    "IT@LAU:4:063049": "coastal",
    "IT@NUTS:3:ITF33": "coastal",
    "ES@LAU:4:08096": "river",
    "DE@NUTS:3:DEB3K": "agroforestry",
}


@pytest.fixture(scope="module")
def eurostat():
    reg = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    return prepare(ObservationStore(str(STORE)), reg, "2018/2022", levels=[0, 2, 3], merge_copies=False)


def _coverage(d, split):
    tr = d.with_mask(split.train(d.mask))
    m = Conformalised(HierTypedMF, seed=split.seed).fit(tr)
    inside = np.abs(m.predict() - d.Z) <= Z90 * m.predict_std()
    lv = d.levels()
    return [
        {"scheme": split.scheme[:2], "level": int(v), "n": int(t.sum()), "hit": int(inside[t].sum())}
        for v in np.unique(lv)
        if (t := split.test & (lv[:, None] == v) & np.isfinite(m.predict_std())).any()
    ]


@needs_build
def test_m5_conformal_coverage(eurostat):
    from joblib import Parallel, delayed

    d = eurostat
    splits = [s for s in prototype_splits(d) if s.scheme.startswith(("M1", "M2"))]
    rows = sum(Parallel(n_jobs=4)(delayed(_coverage)(d, s) for s in splits), [])
    cov = pd.DataFrame(rows).groupby(["scheme", "level"])[["n", "hit"]].sum()
    cov["coverage"] = cov.hit / cov.n
    bad = cov[(cov.coverage < 0.87) | (cov.coverage > 0.93)]
    assert bad.empty, cov.round(3).to_string()


@needs_build
def test_m5_typology_reproducible(eurostat):
    run = TypologyRun(lens=Lens(levels=(2,)), algorithm="gmm", k_range=(3, 8), n_boot=10, seed=0)
    out = []
    for _ in range(2):
        model = HierTypedMF().fit(eurostat)
        out.append(
            run_typology(model, eurostat, run, samples=model.predict_dist(run.n_posterior, noise=False))
        )
    a, b = out
    assert a.config_hash == b.config_hash
    assert np.array_equal(a.labels, b.labels)
    assert a.cluster_table.equals(b.cluster_table)


def test_m5_gates_expressible():
    rules = load_rules()
    ids = sorted(r.id.split("_")[0] for r in rules.rules)
    assert ids == [f"G{i:03d}" for i in range(1, 11)]


@pytest.mark.skipif(not GPKG.exists(), reason="D3.6 GeoPackage not downloaded")
def test_m5_gates_reproduce_published():
    rules = load_rules()
    values = load_values(GPKG)
    for demo, cic in DEMONSTRATORS.items():
        pub = published_gates(GPKG, demo, cic)
        ours = evaluate(rules, values.loc[demo], values.reindex(pub.index), cic=cic, published_only=True)
        assert set(ours.columns) == set(pub.columns) - {"eligible"}, demo
        for g in ours.columns:
            assert (ours[g] == pub[g]).all(), (demo, g)
        assert (gate_summary(ours, rules, mode="d36").eligible == pub.eligible).all(), demo
