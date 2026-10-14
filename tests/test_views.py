import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import shapely  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from regsim.models.hier_mf import HierTypedMF  # noqa: E402
from regsim.schema import Lens  # noqa: E402
from regsim.synthetic import generate_hierarchical  # noqa: E402
from regsim.typology.run import TypologyRun, run_typology, track  # noqa: E402
from regsim.viz import views as V  # noqa: E402

warnings.filterwarnings("ignore")
go = pytest.importorskip("plotly.graph_objects")


@pytest.fixture(scope="module")
def fitted():
    d = generate_hierarchical(
        n_per_level=(4, 20, 80),
        n_indicators=8,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        n_periods=2,
        seed=2,
    )
    md = d.model_data()
    return md, HierTypedMF(rank=3, lam_u=0.3, reconcile=False).fit(md)


def _both(fn, *args, **kwargs):
    static = fn(*args, **kwargs)
    inter = fn(*args, interactive=True, **kwargs)
    assert isinstance(static, Figure) and isinstance(inter, go.Figure)
    return static, inter


def test_embedding_views_report_their_statistics(fitted):
    d, m = fitted
    lv = int(d.levels().max())
    fig, _ = _both(V.biplot, m, d, level=lv)
    assert 0 < sum(fig.regsim_stats["explained_variance"]) <= 1
    fig = V.distortion(m, d, level=lv, methods=("pca",), n_sample=60)
    assert 0 <= fig.regsim_stats["pca"]["trustworthiness"] <= 1
    fig, _ = _both(V.indicator_similarity, m, d, level=1, min_pairs=5)
    assert 0 < fig.regsim_stats["share_observed_pairs"] <= 1


def test_profiles_difference_and_trajectories(fitted):
    d, m = fitted
    lv = int(d.levels().max())
    ty = run_typology(m, d, TypologyRun(lens=Lens(levels=(lv,)), k=3, n_boot=2))
    _both(V.type_profiles, ty)
    assert isinstance(V.consensus_heatmap(ty), Figure)
    _, trans = track(ty, ty)
    _both(V.transitions, trans)
    _both(V.difference_profile, m, d, d.geo_ids[-1], d.geo_ids[-2])
    if d.period is not None:
        bases = sorted({g.split("@")[0] for g in d.geo_ids if d.levels()[d.geo_ids.index(g)] == 1})[:3]
        fig, _ = _both(V.trajectories, m, d, bases, level=1)
        assert fig.regsim_stats["median_displacement"] >= 0


def test_maps_and_benchmark_views():
    geometry = {f"R{i}": shapely.box(3e6 + i * 1e5, 2e6, 3e6 + (i + 1) * 1e5, 2.1e6) for i in range(4)}
    panels = {
        "types": pd.Series([0, 1, 1, 2], index=list(geometry)),
        "score": pd.Series([0.1, 0.5, 0.7, np.nan], index=list(geometry)),
    }
    _both(V.maps, panels, geometry)
    res = pd.DataFrame(
        {
            "scheme": ["M1 random cells"] * 2 + ["M2 cold start (level 2)"] * 4,
            "method": ["A", "B", "A", "A", "B", "B"],
            "k": [np.nan, np.nan, 1, 3, 1, 3],
            "rmse": [0.5, 0.4, 0.9, 0.7, 0.8, 0.6],
        }
    )
    a, b = V.model_card(res)
    assert isinstance(a, Figure) and isinstance(b, Figure)


def test_coverage_and_calibration(fitted):
    d, m = fitted
    assert isinstance(V.coverage_matrix(d), Figure)
    sd = m.predict_std()
    test = d.mask & np.isfinite(sd)
    fig = V.calibration_plot(m.predict(), sd, d.Z, test)
    assert 0.5 < fig.regsim_stats["coverage"] <= 1
