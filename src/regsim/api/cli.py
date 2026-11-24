"""Command-line interface."""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(help="regsim: similarity and prediction for multi-level regional indicators")
causal_app = typer.Typer(help="Causal layer (M7): edge registry, consistency checks, panel estimates")
app.add_typer(causal_app, name="causal")


@causal_app.command("check")
def causal_check(
    config: str = typer.Option(..., "--config", help="run config with a causal section"),
    write_registry: bool = typer.Option(False, help="write E1 items into the edge registry file"),
):
    """C1: sign checks of the edge registry against region means of the annual panels."""
    from regsim.causal.check import run_check
    from regsim.config import RunConfig

    table, rates, folder = run_check(RunConfig.from_yaml(config), write_registry=write_registry)
    typer.echo(table.label.value_counts().to_string())
    typer.echo(rates[rates.grade == "all"].to_string(index=False))
    typer.echo(f"written to {folder}")


@causal_app.command("fit")
def causal_fit(
    config: str = typer.Option(..., "--config", help="run config with a causal section"),
    write_registry: bool = typer.Option(
        False, help="write E2 evidence and temporal precedence into the registry"
    ),
    direction: bool = typer.Option(True, help="run the direction tests"),
):
    """C2: panel effects of all registry edges (pooled and per region group) and direction tests."""
    from regsim.causal.run import run_fit
    from regsim.config import RunConfig

    effects, directions, folder = run_fit(RunConfig.from_yaml(config), write_registry, direction)
    pooled = effects[effects.group == "all"]
    typer.echo(f"{len(pooled)} edges estimated, {int((pooled.t.abs() > 1.96).sum())} with |t| > 1.96")
    if len(directions):
        typer.echo(directions.result.value_counts().to_string())
    typer.echo(f"written to {folder}")


@causal_app.command("what-if")
def causal_what_if(
    geo: str,
    shock: Annotated[
        list[str], typer.Option("--shock", help="indicator=relative change, e.g. gdp_pps_hab=0.1")
    ],
    config: str = typer.Option(..., "--config"),
    horizon: str = typer.Option("short_run", help="short_run or long_run"),
):
    """P8: consequences of relative changes for the descendants of the shocked indicators."""
    from regsim.causal.run import run_what_if
    from regsim.config import RunConfig

    shocks = {k: float(v) for k, v in (s.split("=", 1) for s in shock)}
    out = run_what_if(RunConfig.from_yaml(config), geo, shocks, horizon)
    cols = ["indicator", "value_now", "change_abs", "change", "lower", "upper", "tier", "sources"]
    typer.echo(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


@causal_app.command("effects")
def causal_effects(
    config: str = typer.Option(..., "--config"),
    write_store: bool = typer.Option(
        True, help="store region effects as derived indicators (scenario effect)"
    ),
):
    """M7.6: validation gate and region effects of the links in `causal.effect_links`."""
    from regsim.causal.run import run_effects
    from regsim.config import RunConfig

    gate, table, folder = run_effects(RunConfig.from_yaml(config), write_store)
    typer.echo(gate.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    typer.echo(f"{len(table)} region effects; written to {folder}")


@causal_app.command("compare")
def causal_compare(geo_a: str, geo_b: str, config: str = typer.Option(..., "--config")):
    """Effect similarity of two regions: P(practical equivalence) per link and effect distance."""
    from regsim.causal.run import run_compare
    from regsim.config import RunConfig

    out = run_compare(RunConfig.from_yaml(config), geo_a, geo_b)
    typer.echo(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    typer.echo(f"effect distance: {out.attrs['effect_distance']:.3f}")


@causal_app.command("literature")
def causal_literature(
    edges: Annotated[list[str], typer.Argument(help="registry edge ids, e.g. E018")],
    config: str = typer.Option(..., "--config"),
):
    """M7.2: search, extract, verify, strict measure check and grade for registry edges (LLM from causal.llm)."""
    from regsim.causal.run import run_literature
    from regsim.config import RunConfig

    graded, folder = run_literature(RunConfig.from_yaml(config), edges)
    typer.echo(graded.to_string(index=False))
    typer.echo(f"written to {folder}")


@causal_app.command("sd")
def causal_sd(config: str = typer.Option(..., "--config")):
    """M7.4: SD perturbation protocol, comparison with expectations, lever ranking, domain check."""
    from regsim.causal.run import run_sd
    from regsim.config import RunConfig

    out = run_sd(RunConfig.from_yaml(config))
    review = out["compare"][out["compare"].status == "model_review"]
    typer.echo(
        review[["city", "driver", "output", "elasticity", "reason", "diagnosis"]].to_string(index=False)
    )
    r = out["ranking"]
    typer.echo(f"lever ranking agreement (Spearman): {r.attrs['spearman']}")
    d = out["domain"]
    typer.echo(
        f"domain: median distance {d.distance_nearest.median():.3f}, outside range {d.outside_any.mean():.2f}; "
        f"best next demonstrator {out['next_demonstrator'].candidate.iloc[0]}"
    )
    typer.echo(f"written to {out['folder']}")


@causal_app.command("direction")
def causal_direction(
    a: str,
    b: str,
    config: str = typer.Option(..., "--config"),
    n_tests: int = typer.Option(1, help="number of tests for the Bonferroni threshold"),
):
    """Temporal precedence of two indicators (lagged tests both ways, event profile)."""
    from regsim.causal.panel import direction_test, reference_offsets
    from regsim.causal.run import panel
    from regsim.config import RunConfig

    cfg = RunConfig.from_yaml(config)
    r = direction_test(
        panel(cfg), a, b, n_tests=n_tests, offsets=reference_offsets(cfg.causal.reference_periods)
    )
    typer.echo({k: v for k, v in r.items() if k != "event_profile"})


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
def fit(config: str, period: str = "", refresh: bool = False):
    """Fit the calibrated model of the config (cached in ``<output_dir>/fit-<hash>-<period>``)."""
    from regsim.api.runtime import fit as run_fit
    from regsim.config import RunConfig

    f = run_fit(RunConfig.from_yaml(config), period or None, refresh=refresh)
    typer.echo(f"{f.folder}  entities={f.data.Z.shape[0]}  indicators={f.data.Z.shape[1]}")


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
    from regsim.api.runtime import fit as run_fit
    from regsim.api.runtime import resolve_id
    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, s_profile

    f = run_fit(cfg, period or None)
    chosen = Lens() if lens == "all" else Lens(indicators=tuple(x.strip() for x in lens.split(",")))
    space = s_profile(f.model, f.data, chosen)
    samples = f.base.predict_dist(100, noise=False)
    typer.echo(nearest(space, resolve_id(f.data, geo_id), k=k, samples=samples).to_string(index=False))


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
def predict(config: str, geo_id: str, indicator: str = "", period: str = "", alpha: float = 0.1):
    """Completed values of GEO_ID with calibrated intervals (task P1) and its applicability domain."""
    import pandas as pd

    from regsim.api.runtime import fit as run_fit
    from regsim.api.runtime import resolve_id
    from regsim.config import RunConfig
    from regsim.predict.tasks import fill_gaps
    from regsim.uncertainty.domain import applicability

    f = run_fit(RunConfig.from_yaml(config), period or None)
    geo_id = resolve_id(f.data, geo_id)
    pd.set_option("display.width", 200)
    typer.echo(f"applicability: {applicability(f.model, f.data).loc[geo_id, 'domain']}")
    typer.echo(fill_gaps(f.model, geo_id, indicator or None, alpha=alpha).to_string(index=False))


@app.command()
def report(config: str, out: str = "", only: str = ""):
    """Regenerate the figures of docs/visualisation.md (PNG and interactive HTML) from the config."""
    from regsim.config import RunConfig
    from regsim.viz.report import generate

    folder = generate(
        RunConfig.from_yaml(config), out or None, [x.strip() for x in only.split(",")] if only else None
    )
    typer.echo(f"figures: {folder / 'figures'}")


@app.command(name="export-lau")
def export_lau(
    config: str,
    out: str,
    country: str = "AT",
    indicators: str = "statat_population,statat_share_age_65plus",
    period: str = "",
):
    """Write the LAUs of COUNTRY with INDICATORS (comma-separated) and D3.6 geometries to the GeoPackage OUT."""
    from regsim.config import RunConfig
    from regsim.export import export_lau as run_export

    path, prov = run_export(
        RunConfig.from_yaml(config), out, country, [x.strip() for x in indicators.split(",")], period or None
    )
    typer.echo(f"{path}: {prov['n_units']} units")
    for k, v in prov["indicators"].items():
        typer.echo(f"  {k}: period {v['period']}, missing {v['missing']}, {', '.join(v['source_version'])}")


@app.command()
def serve(config: str, host: str = "127.0.0.1", port: int = 8000):
    """REST service for the config (FastAPI; extra `api`)."""
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit('regsim serve needs FastAPI and uvicorn: pip install -e ".[api]"') from e

    from regsim.api.service import create_app
    from regsim.config import RunConfig

    uvicorn.run(create_app(RunConfig.from_yaml(config)), host=host, port=port)


@app.command(name="app")
def explorer(
    config: str,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    effects: str = typer.Option(
        "", help="causal config whose `regsim causal effects` run feeds the Effects tab"
    ),
):
    """Interactive explorer with linked map and embedding (Dash; extra `app`, decision D7)."""
    from pathlib import Path

    try:
        import dash  # noqa: F401
    except ImportError as e:
        raise SystemExit('regsim app needs Dash: pip install -e ".[app]"') from e

    from regsim.api.runtime import fit as run_fit
    from regsim.api.runtime import fit_panel
    from regsim.app.dash_app import create_dash
    from regsim.config import RunConfig
    from regsim.registry import IndicatorRegistry
    from regsim.store.store import ObservationStore
    from regsim.viz.views import load_geometry

    cfg = RunConfig.from_yaml(config)
    path = Path(cfg.data.raw_dir) / "gisco" / f"NUTS_RG_01M_{cfg.data.nuts_version}_3035.parquet"
    geometry = load_geometry(path) if path.exists() else None
    panel = (lambda: fit_panel(cfg)) if len(cfg.data.periods) > 1 else None
    units = ObservationStore(cfg.data.store).units()
    registry = IndicatorRegistry.from_yaml(*cfg.data.registries)
    effect_data = None
    if effects:
        import pandas as pd

        from regsim.causal.run import load_fits

        ecfg = RunConfig.from_yaml(effects)
        folder = Path(ecfg.output_dir) / f"causal-effects-{ecfg.config_hash()}"
        c = ecfg.causal
        effect_data = {
            "fits": load_fits(ecfg),
            "table": pd.read_csv(folder / "region_effects.csv"),
            "delta": {lk: c.equivalence_delta_by_link.get(lk, c.equivalence_delta) for lk in c.effect_links},
        }
    create_dash(
        run_fit(cfg),
        geometry,
        panel=panel,
        region_names=dict(zip(units.geo_id, units.name, strict=True)),
        indicator_labels={k: e.label for k, e in registry.entries.items() if e.label},
        indicator_sources={k: e.source.connector for k, e in registry.entries.items() if e.source},
        effects=effect_data,
    ).run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
