"""Export of LAU units with indicator values and geometry as a GeoPackage, for use by other tools."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from regsim import __version__
from regsim.config import RunConfig
from regsim.store.store import ObservationStore

LAU_LEVEL = 4


def export_lau(
    cfg: RunConfig,
    out: str | Path,
    country: str,
    indicators: list[str],
    period: str | None = None,
    gpkg: str | Path | None = None,
) -> tuple[Path, dict]:
    """Write the LAUs of ``country`` with the given indicators and KNOWING D3.6 geometries (decision D7 of heatimpact).

    Each indicator takes the values of ``period`` or, if ``None``, of its latest stored period. Returns the
    GeoPackage path and the provenance record, which is also written to ``<out>.provenance.json``.
    """
    import geopandas as gpd

    from regsim.data.d36 import fetch_d36

    store = ObservationStore(cfg.data.store)
    units = store.units()
    units = units[(units.level == LAU_LEVEL) & (units.country == country)]
    if units.empty:
        raise ValueError(f"no LAU units for {country} in {cfg.data.store}; build with data.lau: true")
    obs = store.query(
        "SELECT geo_id, indicator_id, period, value, source, source_version, flags FROM observations "
        "WHERE scenario = 'observed' AND indicator_id IN (SELECT unnest(?))",
        [indicators],
    )
    obs = obs[obs.geo_id.isin(set(units.geo_id))]
    table = units[["geo_id", "name", "area_km2"]].set_index("geo_id")
    flags = pd.Series("", index=table.index)
    meta: dict[str, dict] = {}
    for ind in indicators:
        o = obs[obs.indicator_id == ind]
        if o.empty:
            raise ValueError(f"indicator {ind} has no LAU values for {country}")
        p = period or max(o.period)
        o = o[o.period == p]
        table[ind] = o.groupby("geo_id").value.mean()
        f = o[o["flags"] != ""].groupby("geo_id")["flags"].agg(" ".join)
        flags.loc[f.index] = (flags.loc[f.index] + " " + ind + ":" + f).str.strip()
        meta[ind] = {
            "period": p,
            "source": sorted(o.source.unique()),
            "source_version": sorted(o.source_version.unique()),
            "missing": int(table[ind].isna().sum()),
        }
    table["flags"] = flags

    gpkg_path = Path(cfg.data.d36_gpkg or fetch_d36(f"{cfg.data.raw_dir}/d36"))
    regions = gpd.read_file(
        gpkg_path, layer="knowing.regions", where=f"iso2 = '{country}' AND level = {LAU_LEVEL}"
    )
    scheme = units.scheme.iloc[0]
    regions["geo_id"] = scheme + ":" + regions.iso2 + ":" + regions.code.astype(str)
    geo = regions[["geo_id", "code", "geometry"]].rename(columns={"code": "lau_code"})
    result = geo.merge(table, left_on="geo_id", right_index=True, how="right")
    no_geom = int(result.geometry.isna().sum())
    if no_geom:
        raise ValueError(f"{no_geom} LAU units without geometry in {gpkg_path.name}")
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=regions.crs)
    result = result[["lau_code", "geo_id", "name", "area_km2", *indicators, "flags", "geometry"]]
    result = result.sort_values("lau_code").reset_index(drop=True)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(out, layer="lau", driver="GPKG")
    build_hash = Path(cfg.data.store) / "build.hash"
    prov = {
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regsim_version": __version__,
        "store": str(cfg.data.store),
        "store_build_hash": build_hash.read_text(encoding="utf-8").strip() if build_hash.exists() else "",
        "config_hash": cfg.config_hash(),
        "country": country,
        "lau_scheme": scheme,
        "n_units": len(result),
        "crs": result.crs.to_string(),
        "geometry_source": f"KNOWING D3.6 GeoPackage {gpkg_path.name} (doi:10.5281/zenodo.20376109), layer knowing.regions",
        "indicators": meta,
    }
    Path(str(out) + ".provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return out, prov
