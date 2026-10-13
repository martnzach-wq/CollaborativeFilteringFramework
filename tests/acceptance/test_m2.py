"""M2 acceptance (HANDOVER.md §3; cold-start tolerance adapted on 2026-10-13, see docs/decisions.md).

Reruns the prototype protocol with the prototype's baselines on the Eurostat 2018-2022 matrix of the store and
compares ranking and RMSE with reference/prototype/prototype_benchmark_summary.csv.
"""

from pathlib import Path

import pandas as pd
import pytest

from regsim.evaluation.benchmark import prototype_splits, run_benchmark, summarise
from regsim.models import baselines as B
from regsim.prep.matrix import prepare
from regsim.registry import IndicatorRegistry
from regsim.store.store import ObservationStore

STORE = Path("data/store")
METHODS = {
    "level mean": B.LevelMean,
    "parent carry-down": B.ParentCarryDown,
    "kNN Gower (D3.6-type)": B.KNNGower,
    "SoftImpute": B.SoftImpute,
}
SCHEMES = {  # prototype name -> regsim name, RMSE tolerance
    "M1 random cells": ("M1 random cells", 0.05),
    "M2 cold start (NUTS-2)": ("M2 cold start (level 2)", 0.10),
    "M4 level transfer (NUTS-2 to NUTS-3)": ("M4 level transfer (to level 3)", 0.05),
}


@pytest.fixture(scope="module")
def comparison():
    if not (STORE / "observations").exists():
        pytest.skip("run `regsim build` first")
    reg = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    data = prepare(ObservationStore(str(STORE)), reg, "2018/2022", levels=[0, 2, 3], merge_copies=False)
    ours = summarise(run_benchmark(data, METHODS, prototype_splits(data), n_jobs=4))
    proto = pd.read_csv("reference/prototype/prototype_benchmark_summary.csv")
    proto = proto[proto.method.isin(METHODS)]
    rows = []
    for p_name, (o_name, tol) in SCHEMES.items():
        a = proto[proto.scheme == p_name].set_index("method")["mean"]
        b = ours[ours.scheme == o_name].set_index("method")["mean"]
        for m in METHODS:
            rows.append({"scheme": p_name, "method": m, "prototype": a[m], "regsim": b[m], "tol": tol})
    return pd.DataFrame(rows)


def test_m2_ranking(comparison):
    for scheme, g in comparison.groupby("scheme"):
        rp = g.prototype.round(3).rank(method="min")
        ro = g.regsim.round(3).rank(method="min")
        assert list(rp) == list(ro), f"{scheme}: ranking differs\n{g}"


def test_m2_rmse(comparison):
    diff = (comparison.regsim - comparison.prototype).abs()
    bad = comparison[diff > comparison.tol]
    assert bad.empty, f"RMSE outside tolerance:\n{bad}"
