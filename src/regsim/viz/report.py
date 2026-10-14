"""`regsim report`: regenerate the figures of docs/visualisation.md from a run configuration."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from regsim.config import RunConfig

FIGURES = {
    "biplot": "fig_viz_biplot",
    "indicator_similarity": "fig_viz_indicator_similarity",
    "distortion": "fig_viz_distortion",
    "maps": "fig_viz_maps",
    "type_profiles": "fig_viz_type_profiles",
    "difference_profile": "fig_viz_difference_profile",
    "trajectories": "fig_viz_trajectories",
    "benchmark": "fig_benchmark",
    "coldstart": "fig_coldstart",
}


def _model(cfg: RunConfig):
    from regsim.models.hier_mf import HierTypedMF

    mc = cfg.model
    return lambda: HierTypedMF(
        rank=mc.rank, lam_u=mc.lam_u, lam_v=mc.lam_v, lam_agg=mc.lam_agg, coupling=mc.coupling, seed=mc.seed
    )


def _benchmark(cfg: RunConfig, data, factory) -> pd.DataFrame:
    """Results of `regsim evaluate` if given, else a reduced prototype benchmark (M1 and M2, four methods)."""
    if cfg.report.benchmark:
        return pd.read_csv(cfg.report.benchmark)
    from regsim.evaluation.benchmark import prototype_splits, run_benchmark
    from regsim.models.baselines import LevelMean, ParentCarryDown, SoftImpute

    splits = [
        s for s in prototype_splits(data, m1_seeds=(0,), m2_seeds=(100,)) if s.scheme[:2] in ("M1", "M2")
    ]
    est = {
        "LevelMean": LevelMean,
        "ParentCarryDown": ParentCarryDown,
        "SoftImpute": lambda: SoftImpute(lam=5.0, rank_max=12),
        "HierTypedMF": factory,
    }
    return run_benchmark(data, est, splits)


def generate(cfg: RunConfig, out_dir: str | Path | None = None, only: list[str] | None = None) -> Path:
    """Fit the models of ``cfg``, render every view and write PNG (and HTML) files, the statistics of the views
    and the resolved configuration to ``out_dir`` (default ``<output_dir>/report-<hash>``)."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from regsim.api.runtime import resolve_id, short_id
    from regsim.prep.matrix import prepare
    from regsim.registry import IndicatorRegistry
    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, percentile_score, s_profile
    from regsim.store.store import ObservationStore
    from regsim.typology.run import TypologyRun, run_typology
    from regsim.uncertainty.conformal import Conformalised
    from regsim.uncertainty.domain import applicability
    from regsim.viz import views as V

    rc = cfg.report
    out = Path(out_dir or Path(cfg.output_dir) / f"report-{cfg.config_hash()}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(out)
    want = set(only or FIGURES)
    store = ObservationStore(cfg.data.store)
    registry = IndicatorRegistry.from_yaml(*cfg.data.registries)
    factory = _model(cfg)
    stats: dict = {}

    def save(name, fig, fig_html=None):
        fig.savefig(out / "figures" / f"{FIGURES[name]}.png", dpi=rc.dpi, bbox_inches="tight")
        stats[name] = getattr(fig, "regsim_stats", None)
        plt.close(fig)
        if rc.html and fig_html is not None:
            fig_html.write_html(out / "figures" / f"{FIGURES[name]}.html", include_plotlyjs="cdn")

    def html(fn, *a, **k):
        if not rc.html:
            return None
        try:
            return fn(*a, interactive=True, **k)
        except ImportError:
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = prepare(store, registry, cfg.data.periods[-1], levels=cfg.data.levels)
        model = factory().fit(data)
    lv = data.levels()
    ty = None
    if want & {"biplot", "maps", "type_profiles"}:
        samples = model.predict_dist(10, noise=False)
        run_a = TypologyRun(
            lens=Lens(levels=(rc.level,)), k=rc.typology_k, n_boot=rc.typology_boot, seed=rc.seed
        )
        ty = run_typology(model, data, run_a, samples=samples)
        labels = pd.Series(ty.labels, index=ty.geo_ids)
    if "biplot" in want:
        save(
            "biplot",
            V.biplot(model, data, level=rc.level, colour_by=labels),
            html(V.biplot, model, data, level=rc.level, colour_by=labels),
        )
    if "indicator_similarity" in want:
        save(
            "indicator_similarity",
            V.indicator_similarity(model, data, rc.similarity_level),
            html(V.indicator_similarity, model, data, rc.similarity_level),
        )
    if "distortion" in want:
        save(
            "distortion",
            V.distortion(model, data, level=rc.level, n_sample=rc.n_sample_distortion, seed=rc.seed),
            None,
        )
    if "maps" in want:
        rows = np.flatnonzero(lv == rc.level)
        share = data.mask[rows].mean(axis=0)
        lens_b = Lens(
            indicators=tuple(
                c for c, s in zip(data.column_ids, share, strict=True) if s >= rc.observed_share
            ),
            levels=(rc.level,),
        )
        ty_b = run_typology(
            model,
            data,
            TypologyRun(lens=lens_b, k=rc.typology_k, n_boot=rc.typology_boot, seed=rc.seed),
            samples=samples,
        )
        space = s_profile(model, data)
        reference = resolve_id(data, rc.reference)
        g = data.geo_ids.index(reference)
        d = space.distances(g, rows)
        sim = pd.Series(
            percentile_score(d, space.pair_distribution(rc.level)), index=[data.geo_ids[i] for i in rows]
        )
        obs_share = pd.Series(data.mask[rows].mean(axis=1), index=sim.index)
        geometry = V.load_geometry(
            Path(cfg.data.raw_dir) / "gisco" / "NUTS_RG_01M_2024_3035.parquet", rc.level
        )
        ad = applicability(model, data)
        panels = {
            f"(a) typology, {V.lens_label(data, None, rc.level)}: k={ty.k}, ARI {ty.stability:.2f}": labels,
            f"(b) typology, {len(lens_b.indicators)} indicators observed at NUTS-{rc.level}: k={ty_b.k}, "
            f"ARI {ty_b.stability:.2f}": pd.Series(ty_b.labels, index=ty_b.geo_ids),
            f"(c) similarity to {rc.reference} (percentile)": sim,
            "(d) share of observed indicators": obs_share,
        }
        panels = {k: v.rename(index=short_id) for k, v in panels.items()}
        save("maps", V.maps(panels, geometry, flags=ad.domain.rename(index=short_id)), None)
        stats["maps_typology"] = {
            "a": ty.cluster_table.reset_index().to_dict("records"),
            "b": ty_b.cluster_table.reset_index().to_dict("records"),
        }
    if "type_profiles" in want:
        save("type_profiles", V.type_profiles(ty), html(V.type_profiles, ty))
    if "difference_profile" in want:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = Conformalised(factory, seed=rc.seed).fit(data)
        reference = resolve_id(data, rc.reference)
        partner = (
            resolve_id(data, rc.pair_with)
            if rc.pair_with
            else nearest(s_profile(model, data), reference, k=1).geo_id.iloc[0]
        )
        save(
            "difference_profile",
            V.difference_profile(cm, data, reference, partner),
            html(V.difference_profile, cm, data, reference, partner),
        )
        stats["difference_pair"] = [reference, partner]
    if "trajectories" in want and len(cfg.data.periods) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            levels = sorted({lv for lv in cfg.data.levels if lv <= rc.trajectory_level})
            data_t = prepare(store, registry, cfg.data.periods, levels=levels)
            model_t = factory().fit(data_t)
        bases = {g.split("@")[0] for g in data_t.geo_ids}
        regions = [b for b in sorted(bases) if short_id(b) in set(rc.trajectory_regions)]
        save(
            "trajectories",
            V.trajectories(model_t, data_t, regions, rc.trajectory_level),
            html(V.trajectories, model_t, data_t, regions, rc.trajectory_level),
        )
    if want & {"benchmark", "coldstart"}:
        res = _benchmark(cfg, data, factory)
        res.to_csv(out / "benchmark.csv", index=False)
        save("benchmark", V.benchmark_chart(res), html(V.benchmark_chart, res))
        save("coldstart", V.cold_start_chart(res), html(V.cold_start_chart, res))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    return out
