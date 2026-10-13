"""M3 acceptance (HANDOVER.md §3).

1. Synthetic data (generator defaults, MCAR 30 %): principal angles between true and estimated factor subspaces
   below 15 degrees.
2. Eurostat 2018-2022: RMSE of the tuned HierTypedMF at most the prototype's "MF, prior + aggregation" in M1-M4.
3. Coherence of aggregable indicators: aggregate of children within 1 % of the observed parent value.
4. Fold-in of new entities within 0.02 RMSE of a full refit.

Tests 2-4 need a completed `regsim build`; test 2 takes about 15 minutes on 8 cores.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from regsim.evaluation.benchmark import prototype_splits, run_benchmark, summarise
from regsim.models.hier_mf import HierTypedMF, coherence_error
from regsim.models.tuning import Tuned
from regsim.prep.matrix import prepare
from regsim.registry import IndicatorRegistry
from regsim.store.store import ObservationStore
from regsim.synthetic import generate_hierarchical

warnings.filterwarnings("ignore")
STORE = Path("data/store")
needs_build = pytest.mark.skipif(not (STORE / "observations").exists(), reason="run `regsim build` first")
PROTOTYPE = {  # "MF, prior + aggregation", docs/prototype.md
    "M1 random cells": 0.425,
    "M2 cold start (level 2)": 0.507,
    "M3 country blocks": 0.723,
    "M4 level transfer (to level 3)": 0.410,
}
GRID = {"rank": [8, 12], "lam_u": [0.03, 0.3], "lam_agg": [1.0, 10.0]}


@pytest.fixture(scope="module")
def eurostat():
    reg = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    return prepare(ObservationStore(str(STORE)), reg, "2018/2022", levels=[0, 2, 3], merge_copies=False)


def test_m3_synthetic_subspace_recovery():
    d = generate_hierarchical(rank=5, noise=0.1, missingness="mcar", missing_rate=0.3, seed=0)
    m = HierTypedMF(rank=5, lam_u=0.03, lam_v=0.01, reconcile=False).fit(d.model_data())
    assert np.degrees(subspace_angles(d.V, m.V_)).max() < 15
    assert np.degrees(subspace_angles(d.U - d.U.mean(0), m.U_ - m.U_.mean(0))).max() < 15


@needs_build
def test_m3_rmse_not_worse_than_prototype(eurostat):
    est = {"tuned": lambda: Tuned(HierTypedMF, GRID, n_folds=2)}
    res = summarise(run_benchmark(eurostat, est, prototype_splits(eurostat), n_jobs=8))
    ours = res.set_index("scheme")["mean"]
    bad = {s: (round(float(ours[s]), 3), v) for s, v in PROTOTYPE.items() if ours[s] > v}
    assert not bad, f"RMSE above prototype: {bad}"


@needs_build
def test_m3_coherence(eurostat):
    """Reconciled predictions: 95 % of (parent, column) groups with predicted children within 1 %."""
    d = eurostat
    splits = [s for s in prototype_splits(d) if s.scheme.startswith(("M1", "M4"))]
    for split in splits[:1] + splits[3:5]:
        tr = d.with_mask(split.train(d.mask))
        err = coherence_error(tr, HierTypedMF().fit(tr).predict())
        assert err.size and np.percentile(err, 95) < 0.01, (
            split.scheme,
            split.column,
            np.percentile(err, 95),
        )


@needs_build
def test_m3_fold_in_close_to_refit(eurostat):
    d = eurostat
    rng = np.random.default_rng(7)
    cand = np.flatnonzero((d.level == 3) & (d.mask.sum(axis=1) >= 4))
    rows = rng.choice(cand, int(0.1 * cand.size), replace=False)
    hide = np.zeros_like(d.mask)
    for g in rows:
        obs = np.flatnonzero(d.mask[g])
        hide[g, rng.choice(obs, obs.size // 2, replace=False)] = True
    kw = {"rank": 12, "lam_u": 0.03, "reconcile": False}
    full = HierTypedMF(**kw).fit(d.with_mask(d.mask & ~hide))
    without = d.mask.copy()
    without[rows] = False
    base = HierTypedMF(**kw).fit(d.with_mask(without))
    keep = d.mask[rows] & ~hide[rows]
    F = base.fold_in(np.where(keep, d.Z[rows], 0.0), keep, parent_new=[d.parent[g] for g in rows], rows=rows)

    def rmse(P):
        return float(np.sqrt(np.mean((P - d.Z[rows])[hide[rows]] ** 2)))

    assert abs(rmse(F) - rmse(full.predict()[rows])) < 0.02
