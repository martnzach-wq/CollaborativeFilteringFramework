"""Runs of the causal layer from a config: panels, region groups, panel effects, direction tests.

Results go to ``<output_dir>/causal-<step>-<hash>`` and to the Parquet evidence store (decision D10);
the edge registry file changes only when asked (``write_registry``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from regsim.causal.check import write_evidence
from regsim.causal.graph import EdgeRegistry

EFFECTS_TEST = "panel_c2"
DIRECTION_TEST = "direction"


def _setup(cfg):
    from regsim.config import CausalConfig
    from regsim.registry import IndicatorRegistry
    from regsim.store.store import ObservationStore

    if not (Path(cfg.data.store) / "geo_unit.parquet").exists():
        raise SystemExit(f"no data store in {cfg.data.store!r}: run `regsim build <config>` first")
    c = cfg.causal or CausalConfig()
    return c, ObservationStore(cfg.data.store), IndicatorRegistry.from_yaml(*cfg.data.registries)


def panel(cfg, level: int | None = None) -> pd.DataFrame:
    """Annual panel of the config's store at the causal level (transforms with the causal overrides)."""
    from regsim.causal.data import annual_panel

    c, store, indicators = _setup(cfg)
    since, until = cfg.data.span
    return annual_panel(
        store,
        indicators,
        level=c.level if level is None else level,
        since=since,
        until=until,
        transforms=c.transforms,
    )


def groups(cfg, p: pd.DataFrame, fitted=None) -> pd.Series:
    """Region groups of decision D14 (typology classes of the fitted model, else macro-regions)."""
    from regsim.causal.groups import region_groups

    if fitted is None and cfg.causal is not None and cfg.causal.groups == "typology":
        from regsim.api.runtime import fit

        fitted = fit(cfg)
    return region_groups(cfg, p, fitted)


def run_fit(
    cfg, write_registry: bool = False, direction: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Panel effects of all registry edges (pooled and per group) and direction tests.

    Writes ``effects.csv`` and ``direction.csv`` to ``<output_dir>/causal-fit-<hash>`` and both tables to
    the evidence store. With ``write_registry``, the pooled estimates update ``data_evidence`` and the
    status of each edge (tier E2, `EdgeRegistry.update_evidence`) and the direction results are written
    as ``temporal_precedence``.
    """
    from regsim.causal.effects import estimate_effects
    from regsim.causal.panel import direction_test, reference_offsets

    c, _, indicators = _setup(cfg)
    edges = EdgeRegistry.from_yaml(c.edges)
    p = panel(cfg)
    g = groups(cfg, p)
    kw = {"method": c.estimator, "bias_correction": c.bias_correction}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = estimate_effects(edges, p, g, indicators, c.transforms, **kw)
    effects = effects.assign(group_kind=c.groups)
    pairs = sorted(
        {(e.cause, e.effect) for e in edges.edges if e.cause in p.columns and e.effect in p.columns}
    )
    offsets = reference_offsets(c.reference_periods)
    rows = []
    if direction:
        for a, b in pairs:
            r = direction_test(p, a, b, n_tests=2 * len(pairs), offsets=offsets, events=False)
            rows.append({"cause": a, "effect": b, **{k: v for k, v in r.items() if k != "event_profile"}})
    directions = pd.DataFrame(rows)
    run = cfg.config_hash()
    folder = Path(cfg.output_dir) / f"causal-fit-{run}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(folder)
    effects.to_csv(folder / "effects.csv", index=False)
    directions.to_csv(folder / "direction.csv", index=False)
    write_evidence(effects, cfg.evidence_path, run=run, test=EFFECTS_TEST, tier="E2")
    if len(directions):
        write_evidence(directions, cfg.evidence_path, run=run, test=DIRECTION_TEST, tier="E2")
    if write_registry:
        pooled = effects[effects.group == "all"].rename(columns={"ci95_lo_std": "lo", "ci95_hi_std": "hi"})
        pooled = pooled.assign(
            ci95_std=list(zip(pooled.lo, pooled.hi, strict=True)),
            level=f"{cfg.data.geo_scheme} level {c.level}",
        )
        edges.update_evidence(pooled, tier="E2", test=f"{EFFECTS_TEST} ({c.estimator}, {c.bias_correction})")
        for r in rows:
            for e in edges.edge(r["cause"], r["effect"]):
                e.temporal_precedence = {
                    "result": r["result"],
                    "p_forward": round(float(r["p_forward"]), 6),
                    "p_reverse": round(float(r["p_reverse"]), 6),
                    "test": f"lagged within-region test, lags 1-{r['lags']}, Bonferroni over {2 * len(pairs)} tests",
                }
        edges.to_yaml(c.edges)
    return effects, directions, folder


def load_effects(cfg) -> pd.DataFrame:
    """Panel effects of the config's latest `run_fit` from the evidence store."""
    path = cfg.evidence_path
    run = cfg.config_hash()
    if path.exists():
        ev = pd.read_parquet(path)
        ev = ev[(ev.test == EFFECTS_TEST) & (ev.run == run)]
        if len(ev):
            return ev.dropna(axis=1, how="all").reset_index(drop=True)
    raise SystemExit(
        f"no panel effects for this config in {path}: run `regsim causal fit --config ...` first"
    )


def run_what_if(cfg, geo: str, shocks: dict[str, float], horizon: str = "short_run", alpha: float = 0.1):
    """P8 for ``geo`` with the stored panel effects, the config's region groups and, where
    `regsim causal effects` has run, the entity-specific effects of the effect-similarity model (D12a)."""
    from regsim.api.runtime import fit, resolve_id
    from regsim.causal.queries import what_if

    c, _, _ = _setup(cfg)
    fitted = fit(cfg)
    geo_id = resolve_id(fitted.data, geo)
    effects = load_effects(cfg)
    g = groups(cfg, panel(cfg), fitted)
    fits_path = Path(cfg.output_dir) / f"causal-effects-{cfg.config_hash()}" / "fits.pkl"
    out = what_if(
        fitted,
        EdgeRegistry.from_yaml(c.edges),
        geo_id,
        shocks,
        horizon=horizon,
        alpha=alpha,
        effects=effects,
        groups=g,
        effect_fits=load_fits(cfg) if fits_path.exists() else None,
    )
    return out


def run_effects(cfg, write_store: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Effect similarity (M7.6): validation gate and varying-coefficient fit for each link of
    ``causal.effect_links``, region effects for all entities.

    Writes ``gate.csv``, ``region_effects.csv`` and ``fits.pkl`` to ``<output_dir>/causal-effects-<hash>``,
    the gate to the evidence store and, with ``write_store``, the region effects as derived observations
    (scenario ``effect``) to the observation store.
    """
    import pickle

    from regsim.causal.groups import macro_groups, typology_groups
    from regsim.similarity.effect import effect_observations, fit_varying, moderators, region_effects

    c, store, _ = _setup(cfg)
    p = panel(cfg)
    parents = p.drop_duplicates("geo_id").set_index("geo_id")["parent_id"]
    if c.effect_groups == "typology":
        from regsim.api.runtime import fit

        types = typology_groups(fit(cfg), c.level, seed=cfg.model.seed)
    else:
        types = macro_groups(parents, c.macro_regions)
    fits, gates, effects = {}, [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for link in c.effect_links:
            cause, effect = link.split("->")
            if cause not in p.columns or effect not in p.columns:
                continue
            f = fit_varying(p, cause, effect, moderators(p, effect, types), countries=parents)
            if f is None:
                continue
            fits[link] = f
            g = f.gate
            gates.append(
                {
                    "link": link,
                    "n_groups": g["n_groups"],
                    "alpha": g["alpha"],
                    "rel_err": g["rel_err"],
                    "rel_err_unshrunk": g["rel_err_unshrunk"],
                    "heterogeneous": g["heterogeneous"],
                }
            )
            effects.append(region_effects(f, pd.Index(sorted(f.moderators.index))))
    gate = pd.DataFrame(gates)
    table = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    run = cfg.config_hash()
    folder = Path(cfg.output_dir) / f"causal-effects-{run}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(folder)
    gate.to_csv(folder / "gate.csv", index=False)
    table.to_csv(folder / "region_effects.csv", index=False)
    with (folder / "fits.pkl").open("wb") as fh:
        pickle.dump(fits, fh)
    write_evidence(gate, cfg.evidence_path, run=run, test="effect_gate", tier="E2")
    if write_store and len(table):
        since, until = cfg.data.span
        store.write(effect_observations(table, f"{since}/{until}", run))
    return gate, table, folder


def load_fits(cfg) -> dict:
    """Varying-coefficient fits of the config's latest `run_effects`."""
    import pickle

    path = Path(cfg.output_dir) / f"causal-effects-{cfg.config_hash()}" / "fits.pkl"
    if not path.exists():
        raise SystemExit(f"no effect fits in {path}: run `regsim causal effects --config ...` first")
    with path.open("rb") as fh:
        return pickle.load(fh)


def run_compare(cfg, geo_a: str, geo_b: str) -> pd.DataFrame:
    """Effect similarity of two entities (decision D19 thresholds from the config)."""
    from regsim.similarity.effect import s_effect

    c, store, _ = _setup(cfg)
    fits = load_fits(cfg)
    ids = set(next(iter(fits.values())).moderators.index) if fits else set()

    def resolve(code: str) -> str:
        hits = [g for g in ids if g == code or g.split(":", 1)[-1] == code]
        if len(hits) != 1:
            raise SystemExit(f"unknown entity {code!r} at level {c.level}")
        return hits[0]

    delta = {link: c.equivalence_delta_by_link.get(link, c.equivalence_delta) for link in fits}
    return s_effect(fits, resolve(geo_a), resolve(geo_b), delta=delta)


def load_callable(spec: str):
    """The callable named by ``"module:attribute"`` (decision D15: configurable LLM provider)."""
    import importlib

    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"LLM spec {spec!r} must be 'module:callable'")
    return getattr(importlib.import_module(module), attr)


def run_literature(cfg, edge_ids: list[str], client=None) -> tuple[pd.DataFrame, Path]:
    """Literature pipeline (M7.2) for registry edges: search, extract, verify, strict measure check, grade.

    The LLM callables come from ``causal.llm`` and ``causal.llm_verifier``; the OpenAlex API key from the
    environment variable ``OPENALEX_API_KEY``. Claims, grades and the prompt log go to
    ``<output_dir>/causal-literature-<hash>`` and to the evidence store.
    """
    import os

    from regsim.causal.literature import run_edge, targeted_queries

    c, _, _ = _setup(cfg)
    if not c.llm:
        raise SystemExit("set causal.llm to 'module:callable' (decision D15)")
    llm = load_callable(c.llm)
    verifier = load_callable(c.llm_verifier) if c.llm_verifier else llm
    edges = {e.id: e for e in EdgeRegistry.from_yaml(c.edges).edges}
    grades, claims, logs = [], [], []
    for eid in edge_ids:
        e = edges[eid]
        row = {"id": e.id, "cause": e.cause, "effect": e.effect, "status": e.status}
        qs = targeted_queries(pd.DataFrame([row]))
        res = run_edge(
            pd.Series({**row, "queries": qs, "query": qs[0]}),
            llm,
            verifier,
            client=client,
            api_key=os.environ.get("OPENALEX_API_KEY"),
            max_results=c.literature_max_results,
        )
        grades.append(res["grade"])
        claims.append(res["claims"].assign(edge=eid))
        logs.append(res["log"].assign(edge=eid))
    run = cfg.config_hash()
    folder = Path(cfg.output_dir) / f"causal-literature-{run}"
    folder.mkdir(parents=True, exist_ok=True)
    graded = pd.concat(grades, ignore_index=True)
    graded.to_csv(folder / "grades.csv", index=False)
    pd.concat(claims, ignore_index=True).to_csv(folder / "claims.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(folder / "prompts.csv", index=False)
    write_evidence(graded, cfg.evidence_path, run=run, test="literature", tier="E1")
    return graded, folder


def sd_model_path(cfg) -> Path:
    """Checkout of the KNOWING SD model: ``causal.sd_model_path`` or the environment variable
    ``REGSIM_SD_MODEL``."""
    import os

    path = (cfg.causal.sd_model_path if cfg.causal else None) or os.environ.get("REGSIM_SD_MODEL")
    if not path:
        raise SystemExit("set causal.sd_model_path or REGSIM_SD_MODEL to a checkout of the KNOWING SD model")
    return Path(path)


def lens_matrix(cfg, lens: list[str]) -> pd.DataFrame:
    """Region means of the lens indicators over the config's years (transformed scale), indexed by code
    without scheme prefix."""
    from regsim.api.runtime import short_id

    p = panel(cfg)
    m = p.groupby("geo_id")[[c for c in lens if c in p.columns]].mean()
    m.index = m.index.map(short_id)
    return m


def run_sd(cfg) -> dict:
    """SD bridge (M7.4): perturbation protocol and comparison, lever effects and ranking for the cities of
    ``causal.sd_cities``; domain check and next-demonstrator ranking for ``causal.sd_demonstrators``.

    Writes ``protocol.csv``, ``compare.csv``, ``levers.csv``, ``lever_ranking.csv``, ``domain.csv`` and
    ``next_demonstrator.csv`` to ``<output_dir>/causal-sd-<hash>``; the comparison and lever effects go to
    the evidence store (lever effects are intervention effects for M7.5).
    """
    from regsim.causal import sd_knowing as K
    from regsim.causal.sd_bridge import (
        compare,
        domain_check,
        lever_ranking,
        load_mapping,
        load_parameter_classes,
        next_demonstrator,
    )

    c, _, _ = _setup(cfg)
    model = K.load(sd_model_path(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        protocol = pd.concat(
            [K.perturbation_protocol(model, city) for city in c.sd_cities], ignore_index=True
        )
        levers = pd.concat([K.lever_table(model, city) for city in c.sd_cities], ignore_index=True)
    _, expectations = load_mapping()
    comparison = compare(
        protocol[protocol.year == 2050], EdgeRegistry.from_yaml(c.edges), expectations=expectations
    )
    ranking = lever_ranking(levers)
    lens = load_parameter_classes()["domain_lens"]
    matrix = lens_matrix(cfg, lens)
    domain = domain_check(matrix, None, c.sd_demonstrators, lens)
    candidates = next_demonstrator(matrix.loc[domain.index], c.sd_demonstrators)
    run = cfg.config_hash()
    folder = Path(cfg.output_dir) / f"causal-sd-{run}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(folder)
    for name, t in [
        ("protocol", protocol),
        ("compare", comparison),
        ("levers", levers),
        ("lever_ranking", ranking),
        ("domain", domain.reset_index(names="geo")),
        ("next_demonstrator", candidates),
    ]:
        t.to_csv(folder / f"{name}.csv", index=False)
    write_evidence(
        comparison.astype({"elasticity": float}), cfg.evidence_path, run=run, test="sd_compare", tier="E4"
    )
    write_evidence(  # `effect` names the effect indicator elsewhere in the store
        levers.rename(columns={"effect": "lever_effect"}),
        cfg.evidence_path,
        run=run,
        test="sd_levers",
        tier="E4",
    )
    return {
        "protocol": protocol,
        "compare": comparison,
        "levers": levers,
        "ranking": ranking,
        "domain": domain,
        "next_demonstrator": candidates,
        "folder": folder,
    }


def run_events(
    cfg, path: str = "configs/events.yaml", analogues: int = 20, placebo: int = 200
) -> tuple[pd.DataFrame, Path]:
    """Intervention effects (M7.5) for the events of ``path`` on their outcomes, from the annual panel of
    the config. Events whose post-period has no data yet are reported as pending."""
    from regsim.causal.events import effect, load_events

    events = load_events(path)
    p = panel(cfg)
    rows, summaries = [], []
    for ev in events.itertuples():
        for outcome in ev.outcomes:
            if outcome not in p.columns:
                continue
            one = events[events.id == ev.id]
            out = effect(None, one, outcome, analogues, placebo, cfg.model.seed, panel=p)
            rows.append(out.assign(outcome=outcome))
            s = out.attrs["summary"]
            if "treatment" not in set(s.get("phase", [])):
                s = pd.concat(
                    [s, pd.DataFrame([{"event": ev.id, "geo_id": ev.geo_id, "phase": "treatment",
                                       "status": "pending: no data after the start year"}])],
                    ignore_index=True,
                )  # fmt: skip
            summaries.append(s.assign(outcome=outcome))
    run = cfg.config_hash()
    folder = Path(cfg.output_dir) / f"causal-events-{run}"
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    table.to_csv(folder / "effects.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    if len(summary) and "att" in summary:
        write_evidence(
            summary.dropna(subset=["att"]), cfg.evidence_path, run=run, test="intervention", tier="E3"
        )
    return summary, folder
