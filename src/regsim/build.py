"""Build step: fetch observations, harmonise geo codes to the target NUTS version and fill the store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from regsim.config import RunConfig
from regsim.data.eurostat import EurostatConnector
from regsim.geo.nuts import LauScheme, NutsScheme, correspondence, nuts_table, recode
from regsim.geo.scheme import qualify
from regsim.registry import IndicatorRegistry
from regsim.schema import Aggregation
from regsim.store.store import ObservationStore

log = logging.getLogger(__name__)

POPULATION_ID = "population"
_RECODE = {
    Aggregation.SUM: "sum",
    Aggregation.WMEAN_POP: "wmean",
    Aggregation.WMEAN_AREA: "wmean",
    Aggregation.RATIO: "wmean",
    Aggregation.NONE: "none",
}


@dataclass
class BuildReport:
    units_per_level: dict[int, int] = field(default_factory=dict)
    observations: int = 0
    per_indicator: list[dict] = field(default_factory=list)
    config_hash: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_indicator)


class Harmoniser:
    """Map raw NUTS codes of any supported version to the target version."""

    def __init__(self, target: str, cache_dir: str, sources: tuple[str, ...] = ("2021",)):
        self.target = target
        self.valid = set(nuts_table(target, cache_dir).code)
        self.tables = {s: correspondence(s, target, cache_dir) for s in sources if s != target}

    def harmonise(self, obs: pd.DataFrame, aggregation: Aggregation) -> tuple[pd.DataFrame, dict]:
        """Return observations with target codes in ``geo_code`` and counts of kept, recoded, dropped rows."""
        native = obs[obs.geo_code.isin(self.valid)]
        rest = obs[~obs.geo_code.isin(self.valid)]
        recoded = []
        for table in self.tables.values():
            old = rest[rest.geo_code.isin(set(table.code_from))]
            if old.empty:
                continue
            rest = rest[~rest.index.isin(old.index)]
            how = _RECODE[aggregation]
            for _, grp in old.groupby(["period", *_extra_dims(obs)], dropna=False):
                r = recode(grp.rename(columns={"geo_code": "code"}), table, how)
                if r.empty:
                    continue
                template = grp.iloc[0]
                part = pd.DataFrame(
                    {c: template[c] for c in obs.columns if c not in ("geo_code", "value")},
                    index=range(len(r)),
                )
                part["geo_code"] = r.code.values
                part["value"] = r.value.values
                part["flags"] = (template["flags"] + " recoded").strip()
                recoded.append(part)
        recoded = pd.concat(recoded, ignore_index=True) if recoded else obs.iloc[:0]
        # Values published in the target version win over recoded ones.
        key = ["geo_code", "period", *_extra_dims(obs)]
        recoded = recoded.merge(native[key], on=key, how="left", indicator=True)
        recoded = recoded[recoded._merge == "left_only"].drop(columns="_merge")
        out = pd.concat([native, recoded], ignore_index=True)
        counts = {
            "native": len(native),
            "recoded": len(recoded),
            "dropped": len(rest),
            "dropped_codes": sorted(set(rest.geo_code))[:20],
        }
        return out, counts


def build(config: RunConfig, connector: EurostatConnector | None = None) -> BuildReport:
    """Fetch, harmonise and store observations as described by ``config``."""
    d = config.data
    raw = Path(d.raw_dir)
    gisco = str(raw / "gisco")
    store = ObservationStore(d.store)
    registry = IndicatorRegistry.from_yaml(*d.registries)
    connector = connector or EurostatConnector(cache_dir=str(raw / "eurostat"))
    since, until = d.span
    harmoniser = Harmoniser(d.nuts_version, gisco)
    report = BuildReport(config_hash=config.config_hash())
    scheme_name = f"NUTS{d.nuts_version}"

    # Population (weights and the `population` indicator).
    pw = dict(
        registry.population_weights
        or {"dataset": "demo_r_pjangrp3", "filters": {"sex": "T", "age": "TOTAL", "unit": "NR"}}
    )
    pop_obs = connector.fetch(
        {
            "dataset": pw["dataset"],
            "indicator_id": POPULATION_ID,
            "filters": pw.get("filters", {}),
            "since": since,
            "until": until,
        }
    )
    pop_obs, counts = harmoniser.harmonise(pop_obs, Aggregation.SUM)
    population = pop_obs.groupby("geo_code").value.mean().to_dict()
    frames = [_finish(pop_obs, scheme_name)]
    report.per_indicator.append({"indicator_id": POPULATION_ID, **counts})

    nuts = NutsScheme(d.nuts_version, gisco, population=population, levels=d.levels)
    units, edges = nuts.units(), nuts.edges()
    if d.lau:
        lau = LauScheme(d.nuts_version, cache_dir=gisco)
        units += lau.units()
        edges += lau.edges()
    store.write_geo(units, edges)
    report.units_per_level = pd.Series([u.level for u in units]).value_counts().sort_index().to_dict()

    if "eurostat" in d.sources:
        for entry in registry:
            if entry.source.connector != "eurostat":
                continue
            spec = {
                "dataset": entry.source.dataset,
                "indicator_id": entry.indicator_id,
                "filters": entry.source.filters,
                "since": since,
                "until": until,
            }
            obs = connector.fetch(spec)
            free = [c for c in _extra_dims(obs) if obs[c].nunique() > 1]
            if free:
                raise ValueError(f"{entry.indicator_id}: filters leave free dimensions {free}")
            obs = obs.drop(columns=_extra_dims(obs))
            obs, counts = harmoniser.harmonise(obs, entry.aggregation)
            frames.append(_finish(obs, scheme_name))
            report.per_indicator.append({"indicator_id": entry.indicator_id, **counts})
            log.info("%s: %s", entry.indicator_id, counts)

    if "d36" in d.sources and d.d36_gpkg:
        from regsim.data.d36 import read_d36

        d36 = read_d36(d.d36_gpkg)
        frames.append(d36.observations_for(scheme_name, harmoniser))
        store.write_frame(d36.scores, "d36_precomputed_scores")

    observations = pd.concat(frames, ignore_index=True)
    observations = observations[observations.geo_id.isin({u.geo_id for u in units})]
    store.write(observations)
    report.observations = len(observations)
    store.write_frame(registry_frame(registry), "indicator")
    config.write_resolved(d.store, "build")
    (Path(d.store) / "build_report.json").write_text(
        json.dumps(
            {
                "units_per_level": report.units_per_level,
                "observations": report.observations,
                "per_indicator": report.per_indicator,
                "config_hash": report.config_hash,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return report


def registry_frame(registry: IndicatorRegistry) -> pd.DataFrame:
    canon = registry.canonical()
    return pd.DataFrame(
        [
            {
                "indicator_id": e.indicator_id,
                "label": e.label,
                "dtype": e.dtype.value,
                "transform": e.transform.value,
                "aggregation": e.aggregation.value,
                "native_levels": ",".join(map(str, e.native_levels)),
                "connector": e.source.connector,
                "dataset": e.source.dataset or "",
                "same_as": e.same_as or "",
                "model_indicator": canon[e.indicator_id],
            }
            for e in registry
        ]
    )


def _finish(obs: pd.DataFrame, scheme: str) -> pd.DataFrame:
    out = obs.copy()
    out["geo_id"] = [qualify(scheme, c) for c in out.geo_code]
    out = out.drop(columns=["geo_code"] + _extra_dims(out))
    out["value"] = out.value.astype(float)
    return out[np.isfinite(out.value)]


def _extra_dims(obs: pd.DataFrame) -> list[str]:
    base = {
        "geo_code",
        "geo_id",
        "indicator_id",
        "period",
        "scenario",
        "value",
        "component",
        "se",
        "source",
        "source_version",
        "flags",
    }
    return [c for c in obs.columns if c not in base]
