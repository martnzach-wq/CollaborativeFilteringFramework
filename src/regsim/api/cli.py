"""Command-line interface."""

from __future__ import annotations

import typer

app = typer.Typer(help="regsim: similarity and prediction for multi-level regional indicators")


@app.command()
def build(config: str):
    """Fetch, harmonise and store observations."""
    from regsim.build import build as run_build
    from regsim.config import RunConfig

    report = run_build(RunConfig.from_yaml(config))
    typer.echo(f"units per level: {report.units_per_level}")
    typer.echo(f"observations: {report.observations}")
    typer.echo(report.to_frame()[["indicator_id", "native", "recoded", "dropped"]].to_string(index=False))


@app.command()
def fit(config: str):
    raise NotImplementedError("M3")


@app.command()
def evaluate(config: str):
    """Masked hold-out benchmark of the configured estimators; writes a model card."""
    from regsim.config import RunConfig
    from regsim.evaluation.benchmark import summarise
    from regsim.evaluation.run import evaluate as run_evaluate

    results, card = run_evaluate(RunConfig.from_yaml(config))
    typer.echo(summarise(results).to_string(index=False))
    typer.echo(f"model card: {card}")


@app.command()
def similar(
    config: str,
    geo_id: str,
    k: int = 10,
    lens: str = "all",
    mode: str = "profile",
    cic: str = "urban",
    period: str = "",
):
    """Most similar entities to GEO_ID.

    ``--mode profile`` (default): S-profile on the completed profiles of HierTypedMF fitted with the config's
    model settings; ``--lens`` is ``all`` or a comma-separated list of indicator ids. ``--mode d36``: KNOWING
    D3.6 scoring of all regions against the demonstrator GEO_ID (a D3.6 geocode) for ``--cic``, with the D3.6
    gates (pairs failing a hard gate are not listed; soft gates are shown in ``gates_fired``).
    """
    import pandas as pd

    from regsim.config import RunConfig

    cfg = RunConfig.from_yaml(config)
    pd.set_option("display.width", 200)
    if mode == "d36":
        from regsim.data.d36 import fetch_d36
        from regsim.similarity.d36 import D36Scorer, apply_gates, load_tables, load_values

        gpkg = cfg.data.d36_gpkg or str(fetch_d36(f"{cfg.data.raw_dir}/d36"))
        scores = D36Scorer(load_tables(gpkg)).score(geo_id, cic)
        scores = apply_gates(scores, load_values(gpkg), geo_id, cic)
        scores = scores[scores.eligible]
        cols = [
            "composite_score",
            "transferability_class",
            "tier1_score",
            "tier2_score",
            "tier3_score",
            "tier4_score",
            "coverage_ratio",
            "gates_fired",
        ]
        typer.echo(
            scores.drop(index=geo_id, errors="ignore").nlargest(k, "composite_score")[cols].to_string()
        )
        return
    from regsim.models.hier_mf import HierTypedMF
    from regsim.prep.matrix import prepare
    from regsim.registry import IndicatorRegistry
    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, s_profile
    from regsim.store.store import ObservationStore

    registry = IndicatorRegistry.from_yaml(*cfg.data.registries)
    window = period or cfg.data.periods[-1]
    data = prepare(ObservationStore(cfg.data.store), registry, window, levels=cfg.data.levels)
    mc = cfg.model
    model = HierTypedMF(
        rank=mc.rank, lam_u=mc.lam_u, lam_v=mc.lam_v, lam_agg=mc.lam_agg, coupling=mc.coupling, seed=mc.seed
    ).fit(data)
    chosen = Lens() if lens == "all" else Lens(indicators=tuple(x.strip() for x in lens.split(",")))
    space = s_profile(model, data, chosen)
    typer.echo(
        nearest(space, geo_id, k=k, samples=model.predict_dist(100, noise=False)).to_string(index=False)
    )


@app.command()
def typology(
    config: str,
    lens: str = "all",
    level: int = 3,
    algorithm: str = "gmm",
    k: int = 0,
    period: str = "",
    seed: int = 0,
):
    """Typology of the entities at LEVEL on the completed profiles of HierTypedMF (``--k 0``: chosen by
    stability and BIC). Writes the resolved run, the cluster table and the assignments to the output folder."""
    import json
    from pathlib import Path

    import pandas as pd

    from regsim.config import RunConfig
    from regsim.models.hier_mf import HierTypedMF
    from regsim.prep.matrix import prepare
    from regsim.registry import IndicatorRegistry
    from regsim.schema import Lens
    from regsim.store.store import ObservationStore
    from regsim.typology.run import TypologyRun, run_typology

    cfg = RunConfig.from_yaml(config)
    registry = IndicatorRegistry.from_yaml(*cfg.data.registries)
    data = prepare(
        ObservationStore(cfg.data.store), registry, period or cfg.data.periods[-1], levels=cfg.data.levels
    )
    mc = cfg.model
    model = HierTypedMF(
        rank=mc.rank, lam_u=mc.lam_u, lam_v=mc.lam_v, lam_agg=mc.lam_agg, coupling=mc.coupling, seed=mc.seed
    ).fit(data)
    indicators = None if lens == "all" else tuple(x.strip() for x in lens.split(","))
    run = TypologyRun(
        lens=Lens(indicators=indicators, levels=(level,)), algorithm=algorithm, k=k or None, seed=seed
    )
    ty = run_typology(model, data, run, samples=model.predict_dist(run.n_posterior, noise=False))
    out = Path(cfg.output_dir) / f"typology-{ty.config_hash}"
    out.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(out)
    (out / "run.json").write_text(json.dumps({"config_hash": ty.config_hash, "run": repr(run)}, indent=1))
    ty.to_frame().to_csv(out / "assignments.csv")
    ty.cluster_table.to_csv(out / "clusters.csv")
    ty.selection.to_csv(out / "selection.csv", index=False)
    pd.set_option("display.width", 200)
    typer.echo(f"{out}  k={ty.k}  stability={ty.stability:.2f}")
    typer.echo(ty.cluster_table.to_string())


@app.command()
def predict(config: str, geo_id: str):
    raise NotImplementedError("M3")


@app.command()
def report(config: str):
    raise NotImplementedError("M6")
