"""C1 consistency checks: observed associations against the prior signs of the edge registry.

For every edge with a prior sign and both ends observed, two statistics on region means of the
annual panel (causal_test_notes.md):

1. between-region Spearman correlation of region means (regional level where both indicators
   are observed for enough regions, else national);
2. within-country Spearman correlation of region means after removing country means (regional
   level only).

Each statistic is labelled ``consistent`` or ``contradicting`` by its sign against
``sign_prior``. Agreement supports evidence tier E1 ("association in the direction expected
from graded evidence"); it does not establish a causal effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from regsim.causal.data import region_means
from regsim.causal.graph import DataEvidence, EdgeRegistry


def _spearman(x: pd.Series, y: pd.Series, min_n: int) -> tuple[float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < min_n or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, n
    return float(stats.spearmanr(x[ok], y[ok]).statistic), n


def _within_country(means: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Region means of ``a`` and ``b`` minus their country means, for regions observed in both and
    countries with at least two such regions."""
    d = means[["parent_id", a, b]].dropna()
    d = d[d.groupby("parent_id")[a].transform("size") >= 2]
    return d[[a, b]] - d.groupby("parent_id")[[a, b]].transform("mean")


def check_edges(
    registry: EdgeRegistry,
    regional: pd.DataFrame,
    national: pd.DataFrame | None = None,
    min_years: int = 5,
    min_regions: int = 30,
    min_countries: int = 15,
) -> pd.DataFrame:
    """Sign checks of all edges against regional and national panels.

    Parameters
    ----------
    registry : EdgeRegistry
    regional, national : DataFrame
        Panels from :func:`regsim.causal.data.annual_panel` at the regional level and at level 0.
    min_years : int
        Years an entity needs in a series to enter a region mean.
    min_regions, min_countries : int
        Minimum number of entities for a correlation at the regional and national level.

    Returns
    -------
    DataFrame
        One row per edge: level used, correlations, entity counts, agreement with the prior sign
        and a label (``consistent``, ``contradicting``, ``no prior``, ``not observed``).
    """
    nodes = sorted({n for e in registry.edges for n in (e.cause, e.effect)})
    reg_means = region_means(regional, nodes, min_years)
    nat_means = region_means(national, nodes, min_years) if national is not None else None
    rows = []
    for e in registry.edges:
        row = {
            "edge_id": e.id,
            "cause": e.cause,
            "effect": e.effect,
            "sign_prior": e.sign_prior,
            "grade": e.grade,
            "status": e.status,
            "mechanism_type": e.mechanism_type,
            "level": None,
            "n_between": 0,
            "r_between": np.nan,
            "n_within_country": 0,
            "r_within_country": np.nan,
        }
        both = e.cause in reg_means.columns and e.effect in reg_means.columns
        if both:
            r, n = _spearman(reg_means[e.cause], reg_means[e.effect], min_regions)
            if np.isfinite(r):
                row.update(level="regional", r_between=r, n_between=n)
                w = _within_country(reg_means, e.cause, e.effect)
                rw, nw = _spearman(w[e.cause], w[e.effect], min_regions)
                row.update(r_within_country=rw, n_within_country=nw)
        national_ok = nat_means is not None and {e.cause, e.effect} <= set(nat_means.columns)
        if row["level"] is None and national_ok:
            r, n = _spearman(nat_means[e.cause], nat_means[e.effect], min_countries)
            if np.isfinite(r):
                row.update(level="national", r_between=r, n_between=n)
        rows.append(row)
    out = pd.DataFrame(rows)
    sign = out.sign_prior.map({"+": 1.0, "-": -1.0})
    for k in ("between", "within_country"):
        r = out[f"r_{k}"]
        agree = np.sign(r) == sign
        out[f"agree_{k}"] = agree.where(r.notna() & sign.notna())
    out["label"] = np.select(
        [out.level.isna(), sign.isna(), out.agree_between == True],  # noqa: E712
        ["not observed", "no prior", "consistent"],
        "contradicting",
    )
    return out


def agreement_rates(table: pd.DataFrame, causal_only: bool = False) -> pd.DataFrame:
    """Share of signed edges whose association agrees with the prior, over distinct cause-effect
    pairs, overall and per grade."""
    t = table[table.sign_prior.notna()]
    if causal_only:
        t = t[t.mechanism_type == "causal"]
    t = t.drop_duplicates(["cause", "effect", "sign_prior"])
    out = []
    for grade, g in [("all", t), *t.groupby("grade")]:
        for k in ("between", "within_country"):
            a = g[f"agree_{k}"].dropna()
            out.append(
                {
                    "grade": grade,
                    "test": k,
                    "n": len(a),
                    "agree": int(a.sum()),
                    "rate": a.mean() if len(a) else np.nan,
                }
            )
    return pd.DataFrame(out)


def apply_e1(registry: EdgeRegistry, table: pd.DataFrame, test: str = "c1_region_means") -> list[str]:
    """Write C1 results into the ``data_evidence`` of edges that have no data evidence yet (tier E1);
    edges with panel evidence (E2 or higher) keep it. Returns the ids of the updated edges."""
    done = []
    by_id = {e.id: e for e in registry.edges}
    for r in table[table.level.notna() & table.sign_prior.notna()].itertuples():
        e = by_id[r.edge_id]
        if e.data_evidence is not None:
            continue
        e.data_evidence = DataEvidence(
            test=test,
            level=str(r.level),
            between_r=round(float(r.r_between), 3),
            within_country_r=None if pd.isna(r.r_within_country) else round(float(r.r_within_country), 3),
            status=str(r.label),
            tier="E1",
        )
        done.append(e.id)
    return done


def write_evidence(
    table: pd.DataFrame, path: str | Path, run: str, test: str = "c1_region_means", tier: str = "E1"
) -> Path:
    """Append rows to the Parquet evidence store (decision D10), replacing earlier rows of the same run
    and test. Tests with different columns share the file; absent columns are null."""
    path = Path(path)
    rows = table.assign(run=run, test=test, tier=tier)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~((old.run == run) & (old.test == test))]
        rows = pd.concat([old, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    return path


def run_check(cfg, write_registry: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """C1 on the store of ``cfg``: writes ``c1.csv``, ``c1_rates.csv`` and the resolved config to
    ``<output_dir>/causal-check-<hash>``, appends the rows to the evidence store and, with
    ``write_registry``, writes E1 items into the edge registry file."""
    from regsim.causal.data import annual_panel
    from regsim.config import CausalConfig
    from regsim.registry import IndicatorRegistry
    from regsim.store.store import ObservationStore

    c = cfg.causal or CausalConfig()
    store = ObservationStore(cfg.data.store)
    if not (Path(cfg.data.store) / "geo_unit.parquet").exists():
        raise SystemExit(f"no data store in {cfg.data.store!r}: run `regsim build <config>` first")
    indicators = IndicatorRegistry.from_yaml(*cfg.data.registries)
    registry = EdgeRegistry.from_yaml(c.edges)
    since, until = cfg.data.span
    regional = annual_panel(store, indicators, level=c.level, since=since, until=until)
    national = annual_panel(store, indicators, level=0, since=since, until=until)
    table = check_edges(registry, regional, national, min_years=c.min_years)
    rates = agreement_rates(table)
    folder = Path(cfg.output_dir) / f"causal-check-{cfg.config_hash()}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(folder)
    table.to_csv(folder / "c1.csv", index=False)
    rates.to_csv(folder / "c1_rates.csv", index=False)
    write_evidence(table, cfg.evidence_path, run=cfg.config_hash())
    if write_registry:
        apply_e1(registry, table)
        registry.to_yaml(c.edges)
    return table, rates, folder
