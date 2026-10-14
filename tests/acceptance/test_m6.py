"""M6 acceptance (HANDOVER.md §3).

1. `regsim report` regenerates all figures of docs/visualisation.md from a config.
2. The REST service passes the contract tests on the real data (response models of `regsim.api.service`).
3. The LAU M4 benchmark (NUTS-3 -> LAU level transfer of D3.6 indicators) is reported: results of
   ``runs/lau-benchmark/benchmark.csv`` (written by the LAU benchmark run, docs/progress.md) or, without them,
   a reduced run on one indicator.

Tests 1 and 2 need `regsim build configs/example.yaml` (about 5 minutes); test 3 needs
`regsim build configs/lau.yaml` (about 12 minutes for the reduced run).
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regsim.config import RunConfig

warnings.filterwarnings("ignore")
STORE = Path("data/store")
LAU_STORE = Path("data/store_lau")
LAU_RESULTS = Path("runs/lau-benchmark/benchmark.csv")
needs_build = pytest.mark.skipif(not (STORE / "observations").exists(), reason="run `regsim build` first")


@needs_build
def test_m6_report_regenerates_all_figures(tmp_path):
    from regsim.viz.report import generate

    wanted = set(
        re.findall(r"figures/([a-z_]+)\.png", Path("docs/visualisation.md").read_text(encoding="utf-8"))
    )
    cfg = RunConfig.from_yaml("configs/example.yaml")
    bench = sorted(Path("runs").glob("evaluate-*/benchmark.csv"))
    cfg.report.benchmark = str(bench[-1]) if bench else None
    folder = generate(cfg, tmp_path / "report")
    made = {p.stem for p in (folder / "figures").glob("*.png") if p.stat().st_size > 10_000}
    assert wanted <= made, wanted - made
    assert (folder / "config.resolved.yaml").exists() and (folder / "stats.json").exists()


@needs_build
def test_m6_rest_contract_on_real_data():
    from fastapi.testclient import TestClient

    from regsim.api import service
    from regsim.api.runtime import fit

    cfg = RunConfig.from_yaml("configs/example.yaml")
    c = TestClient(service.create_app(cfg, fit(cfg)))
    service.Health.model_validate(c.get("/health").json())
    p = c.get("/predict/AT130")
    assert p.status_code == 200
    pred = service.Prediction.model_validate(p.json())
    assert pred.geo_id.endswith("AT130") and pred.domain in {"in_domain", "edge", "outside"}
    s = service.Similar.model_validate(c.get("/similar/AT130", params={"k": 5}).json())
    assert len(s.neighbours) == 5
    v = c.post("/vision", json={"base": "AT130", "targets": {"tertiary_attainment": 60.0}})
    assert v.status_code == 200
    service.Vision.model_validate(v.json())
    t = c.post("/typology", json={"level": 2, "k": 5, "n_boot": 3})
    assert service.Typology.model_validate(t.json()).k == 5
    assert c.get("/predict/XX000").status_code == 404


@pytest.mark.skipif(not (LAU_STORE / "observations").exists(), reason="run `regsim build configs/lau.yaml`")
def test_m6_lau_benchmark_reported():
    if LAU_RESULTS.exists():
        res = pd.read_csv(LAU_RESULTS)
    else:
        from regsim.api.runtime import prepare_data
        from regsim.evaluation.benchmark import level_transfer_benchmark
        from regsim.models.baselines import ParentCarryDown
        from regsim.models.hier_mf import HierTypedMF

        data = prepare_data(RunConfig.from_yaml("configs/lau.yaml"))
        est = {"ParentCarryDown": ParentCarryDown, "HierTypedMF": HierTypedMF}
        res = level_transfer_benchmark(data, ["d36_heat_days_1990_2020_mean"], est, fine_level=4)
    assert {"ParentCarryDown", "HierTypedMF"} <= set(res.method)
    assert res.scheme.str.contains("level 4").all() and (res.n_cells > 10_000).all()
    assert np.isfinite(res.rmse).all() and np.isfinite(res.skill).all()
