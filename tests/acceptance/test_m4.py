"""M4 acceptance (HANDOVER.md §3).

1. D3.6 mode reproduces the urban T2 tier score of the published ``precomputed_scores`` for the Tallinn
   demonstrator, and the composite within the deviation documented in ``docs/data_sources.md`` (3-9 points):
   median absolute deviation at most 3 points, 95th percentile at most 9 points.
2. Neighbour-prediction skill of S-profile at least that of S-gower on scheme M2 (cold start at NUTS 2).

Test 1 needs the D3.6 GeoPackage, test 2 a completed `regsim build` (about 4 minutes).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from regsim.evaluation.masking import entity_cold_start
from regsim.evaluation.neighbours import neighbour_skill
from regsim.models.hier_mf import HierTypedMF
from regsim.prep.matrix import prepare
from regsim.registry import IndicatorRegistry
from regsim.similarity.d36 import D36Scorer, load_tables, published_scores
from regsim.store.store import ObservationStore

warnings.filterwarnings("ignore")
STORE = Path("data/store")
GPKG = Path("data/raw/d36/knowing-transferability-export-v2.5.gpkg")
TALLINN = "EE@LAU:4:0784"


@pytest.mark.skipif(not GPKG.exists(), reason="D3.6 GeoPackage not downloaded")
def test_m4_d36_tallinn_urban():
    pub = published_scores(GPKG, TALLINN, target_level=3)
    pub = pub[pub.cic_context == "urban"]
    assert len(pub) > 1000
    ours = D36Scorer(load_tables(GPKG, regions=[TALLINN], levels=[3])).score(
        TALLINN, "urban", targets=list(pub.index)
    )
    # The published table leaves the tier score of an empty tier NaN but uses distance 1 in the composite.
    published_t2 = pub.tier2_score.to_numpy(dtype=float)
    empty = np.isnan(published_t2)
    assert empty.sum() <= 5 and np.all(ours.tier2_score.to_numpy()[empty] == 0)
    assert np.max(np.abs(ours.tier2_score.to_numpy()[~empty] - published_t2[~empty])) < 1e-6
    dev = np.abs(ours.composite_score.to_numpy() - pub.composite_score.to_numpy())
    assert np.median(dev) <= 3 and np.percentile(dev, 95) <= 9, (np.median(dev), np.percentile(dev, 95))


@pytest.mark.skipif(not (STORE / "observations").exists(), reason="run `regsim build` first")
def test_m4_neighbour_skill_profile_vs_gower():
    reg = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    d = prepare(ObservationStore(str(STORE)), reg, "2018/2022", levels=[0, 2, 3], merge_copies=False)
    splits = [s for seed in (100, 101) for s in entity_cold_start(d.mask, d.level, 2, seed=seed)]
    res = neighbour_skill(d, splits, HierTypedMF)
    rmse = res.groupby("measure").rmse.mean()
    assert rmse["S-profile"] <= rmse["S-gower"], rmse.to_dict()
