"""Adapter for the KNOWING D3.6 GeoPackage (doi:10.5281/zenodo.20376109).

The export holds ``knowing.indicator`` (metadata), ``knowing.indicator_values`` (one row per region, indicator,
period and scenario), ``knowing.regions`` (NUTS-0 to LAU with parent codes) and ``knowing.precomputed_scores``
(Gower-type scores per demonstrator). Region codes have the form ``AT@NUTS:3:AT130`` and ``EE@LAU:4:0784``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regsim.registry import IndicatorEntry, SourceSpec
from regsim.schema import Aggregation, DType, Transform

SOURCE = "d36"
PREFIX = "d36_"

# Indicators whose definition matches an indicator of the Eurostat registry (decision D5).
SAME_AS = {"population_density": "pop_density"}

_DTYPE = {
    "continuous": DType.CONTINUOUS,
    "categorical": DType.CATEGORICAL,
    "ordinal": DType.ORDINAL,
    "compositional": DType.COMPOSITIONAL,
}


@dataclass
class D36Data:
    values: (
        pd.DataFrame
    )  # region_geocode, indicator_id, period, scenario, value_numeric, value_text, value_array
    regions: pd.DataFrame  # geocode, iso2, code, level, name, geocode_parent, typologies, area
    indicators: pd.DataFrame  # knowing.indicator
    scores: pd.DataFrame  # knowing.precomputed_scores
    version: str

    @property
    def n_value_rows(self) -> int:
        return len(self.values)

    def registry_entries(self) -> list[IndicatorEntry]:
        """Registry entries for the populated indicators, with categories taken from the data."""
        populated = set(self.values.indicator_id)
        entries = []
        for r in self.indicators.itertuples():
            if r.id not in populated:
                continue
            dtype = _DTYPE[r.data_type]
            cats = sorted(self.values.loc[self.values.indicator_id == r.id, "value_text"].dropna().unique())
            entries.append(
                IndicatorEntry(
                    indicator_id=PREFIX + r.id,
                    label=r.name,
                    dtype=dtype,
                    transform=_transform(r),
                    aggregation=_aggregation(r.id, dtype),
                    native_levels=[0, 1, 2, 3, 4],
                    unit=r.units or "",
                    categories=[str(c) for c in cats] if dtype in (DType.CATEGORICAL, DType.ORDINAL) else [],
                    note=f"D3.6 theme '{r.theme}' (metadata only); normalisation {r.normalisation}, "
                    f"distance {r.distance_func}",
                    source=SourceSpec(connector=SOURCE, dataset=r.id, version=self.version),
                    same_as=SAME_AS.get(r.id),
                    temporal="static",
                )
            )
        return entries

    def observations(
        self, nuts_scheme: str = "NUTS2024", lau_scheme: str = "LAU2024", valid_nuts: set[str] | None = None
    ) -> tuple[pd.DataFrame, dict]:
        """Observations in store format; categorical values become integer codes into ``categories``.

        Returns the observations and counts: ``rows`` (value rows read), ``stored``, ``empty`` (no value),
        ``unmapped`` (region outside the target schemes) and ``array`` (compositional rows, one per component).
        """
        v = self.values
        entries = {e.indicator_id: e for e in self.registry_entries()}
        geo = _geo_ids(self.regions, nuts_scheme, lau_scheme, valid_nuts)
        empty = v.value_numeric.isna() & v.value_text.isna() & v.value_array.isna()
        frames = []
        num = v[v.value_numeric.notna()]
        frames.append(num.assign(value=num.value_numeric.astype(float), component=None))
        txt = v[v.value_numeric.isna() & v.value_text.notna()]
        if len(txt):
            codes = {
                k: {c: i for i, c in enumerate(entries[PREFIX + k].categories)}
                for k in txt.indicator_id.unique()
            }
            frames.append(
                txt.assign(
                    value=[float(codes[k][t]) for k, t in zip(txt.indicator_id, txt.value_text, strict=True)],
                    component=None,
                )
            )
        arr = v[v.value_numeric.isna() & v.value_text.isna() & v.value_array.notna()]
        n_array = 0
        if len(arr):
            rows = []
            for r in arr.itertuples():
                parts = _array(r.value_array)
                n_array += 1
                rows += [
                    (r.region_geocode, r.indicator_id, r.period, r.scenario, float(x), str(i))
                    for i, x in enumerate(parts)
                ]
            frames.append(
                pd.DataFrame(
                    rows,
                    columns=["region_geocode", "indicator_id", "period", "scenario", "value", "component"],
                )
            )
        obs = pd.concat(frames, ignore_index=True)
        obs["geo_id"] = obs.region_geocode.map(geo)
        unmapped = obs.geo_id.isna()
        out = pd.DataFrame(
            {
                "geo_id": obs.geo_id[~unmapped].values,
                "indicator_id": PREFIX + obs.indicator_id[~unmapped].values,
                "period": obs.period[~unmapped].values,
                # the D3.6 "baseline" is the present state, i.e. observed values; RCP scenarios keep their names
                "scenario": obs.scenario[~unmapped].replace({"baseline": "observed"}).values,
                "value": obs.value[~unmapped].values,
                "component": obs.component[~unmapped].values,
                "se": np.nan,
                "source": SOURCE,
                "source_version": self.version,
                "flags": "",
            }
        )
        counts = {
            "rows": len(v),
            "stored": len(out),
            "empty": int(empty.sum()),
            "unmapped": int(unmapped.sum()),
            "array": n_array,
        }
        return out, counts


ZENODO_RECORD = "20376109"
ZENODO_FILE = "knowing-transferability-export-v2.5.gpkg.zip"


def fetch_d36(raw_dir: str = "data/raw/d36") -> Path:
    """Download and unzip the D3.6 export from Zenodo (skipped when the GeoPackage exists)."""
    import zipfile

    from regsim.data.files import download

    folder = Path(raw_dir)
    gpkg = folder / ZENODO_FILE.removesuffix(".zip")
    if gpkg.exists():
        return gpkg
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZENODO_FILE}/content"
    archive = download(url, folder / ZENODO_FILE)
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
    return gpkg


def read_d36(gpkg_path: str | Path, with_scores: bool = True) -> D36Data:
    """Read the D3.6 GeoPackage tables needed by regsim."""
    path = Path(gpkg_path)
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        values = pd.read_sql(
            "SELECT region_geocode, indicator_id, period_start, period_end, rcp, value_numeric, value_text, "
            'value_array FROM "knowing.indicator_values"',
            con,
        )
        regions = pd.read_sql(
            "SELECT geocode, iso2, code, level, name_latin AS name, geocode_parent, mountain_type, urban_type, "
            'coastal_type, degurba, area FROM "knowing.regions"',
            con,
        )
        indicators = pd.read_sql('SELECT * FROM "knowing.indicator"', con)
        scores = (
            pd.read_sql('SELECT * FROM "knowing.precomputed_scores"', con) if with_scores else pd.DataFrame()
        )
    finally:
        con.close()
    values["period"] = _period(values.period_start, values.period_end)
    values = values.rename(columns={"rcp": "scenario"}).drop(columns=["period_start", "period_end"])
    version = path.stem.rsplit("-", 1)[-1].removeprefix("v") if "-v" in path.stem else path.stem
    return D36Data(values=values, regions=regions, indicators=indicators, scores=scores, version=version)


def write_registry(data: D36Data, path: str | Path) -> Path:
    """Write the D3.6 registry entries as YAML (``configs/indicators_d36_v<version>.yaml``)."""
    import yaml

    doc = {
        "registry_version": f"d36_v{data.version}",
        "indicators": [
            e.model_dump(mode="json", exclude_defaults=True) | {"dtype": e.dtype.value}
            for e in data.registry_entries()
        ],
    }
    path = Path(path)
    path.write_text(
        "# Generated from the KNOWING D3.6 GeoPackage by regsim.data.d36.write_registry; populated indicators only.\n"
        + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _period(start: pd.Series, end: pd.Series) -> pd.Series:
    s, e = start.str[:4], end.str[:4]
    return pd.Series(np.where(s == e, s, s + "/" + e), index=start.index)


def _geo_ids(
    regions: pd.DataFrame, nuts_scheme: str, lau_scheme: str, valid_nuts: set[str] | None
) -> pd.Series:
    nuts = regions.level < 4
    ids = pd.Series(index=regions.geocode, dtype=object)
    codes = regions.code.where(nuts)
    ok = nuts & (codes.isin(valid_nuts) if valid_nuts is not None else True)
    ids[regions.geocode[ok].values] = (nuts_scheme + ":" + regions.code[ok]).values
    lau = regions.level == 4
    ids[regions.geocode[lau].values] = (lau_scheme + ":" + regions.iso2[lau] + ":" + regions.code[lau]).values
    return ids


def _transform(r) -> Transform:
    if r.data_type == "compositional":
        return Transform.CLR
    if r.data_type == "ordinal":
        return Transform.RANK
    # counts of days per year are right-skewed with many small values
    skewed = (r.units or "").strip() == "days/year"
    return Transform.LOG1P if r.log_transform or skewed else Transform.IDENTITY


def _aggregation(indicator_id: str, dtype: DType) -> Aggregation:
    if dtype is not DType.CONTINUOUS:
        return Aggregation.NONE
    if indicator_id == "total_area":
        return Aggregation.SUM
    # electricity_total_demand_mean is an area-weighted mean despite its name: NUTS-3 value = area-weighted mean
    # of the LAU values (median ratio 1.001)
    if indicator_id == "population_density":
        return Aggregation.RATIO
    if "_max" in indicator_id or "_min" in indicator_id or indicator_id.endswith("majority"):
        return Aggregation.NONE
    return Aggregation.WMEAN_AREA


def _array(text) -> list[float]:
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    s = str(text).strip()
    if s.startswith("{") and s.endswith("}"):
        s = "[" + s[1:-1] + "]"
    return [float(x) for x in json.loads(s)]
