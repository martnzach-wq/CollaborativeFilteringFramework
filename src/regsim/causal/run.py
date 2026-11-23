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
            ci95_std=list(zip(pooled.lo, pooled.hi, strict=True)), level=f"level {c.level}"
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
    """P8 for ``geo`` with the stored panel effects and the config's region groups."""
    from regsim.api.runtime import fit, resolve_id
    from regsim.causal.queries import what_if

    c, _, _ = _setup(cfg)
    fitted = fit(cfg)
    geo_id = resolve_id(fitted.data, geo)
    effects = load_effects(cfg)
    g = groups(cfg, panel(cfg), fitted)
    out = what_if(
        fitted,
        EdgeRegistry.from_yaml(c.edges),
        geo_id,
        shocks,
        horizon=horizon,
        alpha=alpha,
        effects=effects,
        groups=g,
    )
    return out
