"""NUTS and LAU schemes from Eurostat GISCO, and correspondence between NUTS versions.

Sources (downloaded once into ``cache_dir``):

- GISCO NUTS regions, 1:1 million, EPSG:3035 (codes, names, typologies, area from geometry);
- Eurostat LAU–NUTS correspondence lists (LAU code, NUTS-3 code, population, area per LAU);
- Eurostat NUTS version change tables (e.g. ``NUTS2021-NUTS2024.xlsx``).

Version correspondence: codes that are unchanged, renamed or only recoded map 1:1. For boundary shifts,
merges, splits and new regions the weights are population overlaps of LAUs listed under both versions,
aggregated from NUTS-3 to the coarser levels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from regsim.data.files import download
from regsim.geo.scheme import qualify
from regsim.schema import GeoEdge, GeoUnit

GISCO = "https://gisco-services.ec.europa.eu/distribution/v2/"
ESTAT_DOCS = "https://ec.europa.eu/eurostat/documents/345175/"
LAU_LEVEL = 4
ONE_TO_ONE = {None, "Name change", "Code change", "art. 5(2a)"}
_LAU_LISTS = {"2021": "501971/EU-27-LAU-2021-NUTS-2021.xlsx", "2024": "501971/EU-27-LAU-2024-NUTS-2024.xlsx"}
_CHANGE_TABLES = {("2021", "2024"): "629341/NUTS2021-NUTS2024.xlsx"}


# ---------------------------------------------------------------------------------------------- tables
def nuts_table(version: str = "2024", cache_dir: str = "data/raw/gisco") -> pd.DataFrame:
    """NUTS regions of one version: ``code, level, country, name, area_km2, mount_type, urbn_type, coast_type``.

    Area is computed from the 1:1 million geometry in EPSG:3035 and needs ``shapely`` (extra ``geo``); the
    result is cached as ``nuts_{version}.parquet``.
    """
    cache = Path(cache_dir) / f"nuts_{version}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import shapely

    src = download(
        GISCO + f"nuts/parquet/NUTS_RG_01M_{version}_3035.parquet",
        Path(cache_dir) / f"NUTS_RG_01M_{version}_3035.parquet",
    )
    raw = pd.read_parquet(src)
    table = pd.DataFrame(
        {
            "code": raw["NUTS_ID"].astype(str),
            "level": raw["LEVL_CODE"].astype(int),
            "country": raw["CNTR_CODE"].astype(str),
            "name": raw["NAME_LATN"].astype(str),
            "area_km2": shapely.area(shapely.from_wkb(raw["Shape"].values)) / 1e6,
            "mount_type": raw["MOUNT_TYPE"].astype("Int64"),
            "urbn_type": raw["URBN_TYPE"].astype("Int64"),
            "coast_type": raw["COAST_TYPE"].astype("Int64"),
        }
    ).sort_values(["level", "code"], ignore_index=True)
    table.to_parquet(cache)
    return table


def lau_table(version: str = "2024", cache_dir: str = "data/raw/gisco") -> pd.DataFrame:
    """LAU list of one year: ``lau_code, country, nuts3, name, population, area_km2, degurba, coastal``."""
    cache = Path(cache_dir) / f"lau_{version}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    src = download(ESTAT_DOCS + _LAU_LISTS[version], Path(cache_dir) / Path(_LAU_LISTS[version]).name)
    book = pd.ExcelFile(src)
    frames = []
    for sheet in book.sheet_names:
        if len(sheet) != 2 or not sheet.isupper():
            continue
        df = pd.read_excel(book, sheet_name=sheet, dtype={"LAU CODE": str})
        df.columns = [str(c).strip().upper().replace("_", " ") for c in df.columns]
        nuts_col = "NUTS3" if "NUTS3" in df.columns else "NUTS 3 CODE"
        df = df[df["LAU CODE"].notna() & df[nuts_col].notna()]
        frames.append(
            pd.DataFrame(
                {
                    "lau_code": df["LAU CODE"].astype(str).str.strip(),
                    "country": sheet,
                    "nuts3": df[nuts_col].astype(str).str.strip(),
                    "name": df["LAU NAME LATIN"].astype(str),
                    "population": pd.to_numeric(df["POPULATION"], errors="coerce"),
                    "area_km2": pd.to_numeric(df["TOTAL AREA (M2)"], errors="coerce") / 1e6,
                    "degurba": pd.to_numeric(df["DEGURBA"], errors="coerce").astype("Int64"),
                    "coastal": df[[c for c in df.columns if c.startswith("COASTAL")][0]].astype(str),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True).drop_duplicates(["country", "lau_code"])
    table.to_parquet(cache)
    return table


def change_table(source: str, target: str, cache_dir: str = "data/raw/gisco") -> pd.DataFrame:
    """Official code changes between two NUTS versions: ``code_from, code_to, level, change``."""
    cache = Path(cache_dir) / f"nuts_changes_{source}_{target}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    name = _CHANGE_TABLES[(source, target)]
    src = download(ESTAT_DOCS + name, Path(cache_dir) / Path(name).name)
    sheet = f"NUTS{source}- NUTS{target}"
    raw = pd.read_excel(src, sheet_name=sheet)
    table = pd.DataFrame(
        {
            "code_from": raw[f"Code {source}"].astype("string"),
            "code_to": raw[f"Code {target}"].astype("string"),
            "level": raw["NUTS level"].astype(int),
            "change": raw["Change"].astype("string"),
        }
    ).dropna(subset=["code_from", "code_to"], how="all")
    table.to_parquet(cache)
    return table


def correspondence(source: str, target: str, cache_dir: str = "data/raw/gisco") -> pd.DataFrame:
    """Population-weighted correspondence between NUTS versions.

    Unchanged, renamed and recoded regions map 1:1. For the other regions of a country, NUTS-3 pieces are
    population overlaps of LAUs listed under both versions when these cover at least ``MIN_LAU_COVERAGE`` of
    the population of every changed region; otherwise the pieces are polygon overlaps of the two versions,
    weighted by the source population. Coarser levels aggregate the NUTS-3 pieces.

    Returns
    -------
    pandas.DataFrame
        ``code_from, code_to, level, w_from, w_to, method``: ``w_from`` is the share of the source region's
        population that lies in the target region (for sums), ``w_to`` the share of the target region's
        population that comes from the source region (for weighted means). ``method`` is ``identity``,
        ``lau_overlap`` or ``polygon_overlap``.
    """
    ch = change_table(source, target, cache_dir)
    is_one = (
        ch.code_from.notna() & ch.code_to.notna() & (ch.change.isna() | ch.change.isin(ONE_TO_ONE - {None}))
    )
    one = ch[is_one]
    changed_from = set(ch.code_from.dropna()) - set(one.code_from)
    changed_to = set(ch.code_to.dropna()) - set(one.code_to)
    ident = pd.DataFrame(
        {
            "code_from": one.code_from,
            "code_to": one.code_to,
            "level": one.level,
            "w_from": 1.0,
            "w_to": 1.0,
            "method": "identity",
        }
    )
    countries = sorted({c[:2] for c in changed_from | changed_to})
    pieces = [_nuts3_pieces(source, target, cc, one, changed_from, changed_to, cache_dir) for cc in countries]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return ident.reset_index(drop=True)
    pieces = pd.concat(pieces, ignore_index=True)
    rows = []
    for level in (3, 2, 1, 0):
        n = level + 2
        p = pieces.assign(code_from=pieces.from3.str[:n], code_to=pieces.to3.str[:n])
        p = p[p.code_from.isin(changed_from) | p.code_to.isin(changed_to)]
        if p.empty:
            continue
        g = p.groupby(["code_from", "code_to"]).agg(
            population=("population", "sum"), method=("method", _method)
        )
        g = g.reset_index()
        g["w_from"] = g.population / g.groupby("code_from").population.transform("sum")
        g["w_to"] = g.population / g.groupby("code_to").population.transform("sum")
        g["level"] = level
        g.loc[(g.method == "identity") & ((g.w_from < 1) | (g.w_to < 1)), "method"] = "nuts3_aggregate"
        rows.append(g.drop(columns="population"))
    changed = pd.concat(rows, ignore_index=True)
    ident = ident[~ident.code_from.isin(changed.code_from) & ~ident.code_to.isin(changed.code_to)]
    return pd.concat([ident, changed], ignore_index=True)


def _method(methods: pd.Series) -> str:
    other = [m for m in methods if m != "identity"]
    return other[0] if other else "identity"


MIN_LAU_COVERAGE = 0.95
MIN_POLYGON_SHARE = 0.005


def _nuts3_pieces(source, target, country, one, changed_from, changed_to, cache_dir) -> pd.DataFrame:
    """NUTS-3 pieces ``from3, to3, population, method`` for one country with changes."""
    a = lau_table(source, cache_dir)
    a = a[a.country == country][["lau_code", "nuts3", "population"]]
    b = lau_table(target, cache_dir)
    b = b[b.country == country][["lau_code", "nuts3", "population"]]
    pop_from = a.groupby("nuts3").population.sum()
    m = a.merge(b, on="lau_code", suffixes=("_from", "_to"))
    from3 = {c for c in changed_from if len(c) == 5 and c.startswith(country)}
    to3 = {c for c in changed_to if len(c) == 5 and c.startswith(country)}
    one3 = one[(one.level == 3) & one.code_from.str.startswith(country)]
    identity = pd.DataFrame(
        {
            "from3": one3.code_from.values,
            "to3": one3.code_to.values,
            "population": [float(pop_from.get(c, np.nan)) for c in one3.code_from],
            "method": "identity",
        }
    )
    cov_from = m.groupby("nuts3_from").population_from.sum() / pop_from
    cov_to = m.groupby("nuts3_to").population_to.sum() / b.groupby("nuts3").population.sum()
    covered = all(cov_from.get(c, 0) >= MIN_LAU_COVERAGE for c in from3) and all(
        cov_to.get(c, 0) >= MIN_LAU_COVERAGE for c in to3
    )
    if covered:
        sub = m[m.nuts3_from.isin(from3) | m.nuts3_to.isin(to3)]
        g = sub.groupby(["nuts3_from", "nuts3_to"]).population_to.sum().reset_index()
        changed = pd.DataFrame(
            {"from3": g.nuts3_from, "to3": g.nuts3_to, "population": g.population_to, "method": "lau_overlap"}
        )
    else:
        changed = _polygon_pieces(source, target, from3, to3, pop_from, cache_dir)
    return pd.concat([identity, changed], ignore_index=True)


def _polygon_pieces(source, target, from3, to3, pop_from, cache_dir) -> pd.DataFrame:
    import shapely

    def geoms(version, codes):
        src = download(
            GISCO + f"nuts/parquet/NUTS_RG_01M_{version}_3035.parquet",
            Path(cache_dir) / f"NUTS_RG_01M_{version}_3035.parquet",
        )
        raw = pd.read_parquet(src, columns=["NUTS_ID", "Shape"])
        raw = raw[raw.NUTS_ID.isin(codes)]
        return dict(zip(raw.NUTS_ID, shapely.from_wkb(raw.Shape.values), strict=True))

    ga, gb = geoms(source, from3), geoms(target, to3)
    rows = []
    for fa, pa in ga.items():
        shares = {fb: shapely.area(shapely.intersection(pa, pb)) / shapely.area(pa) for fb, pb in gb.items()}
        shares = {k: v for k, v in shares.items() if v >= MIN_POLYGON_SHARE}
        total = sum(shares.values())
        for fb, v in shares.items():
            rows.append((fa, fb, float(pop_from.get(fa, np.nan)) * v / total, "polygon_overlap"))
    return pd.DataFrame(rows, columns=["from3", "to3", "population", "method"])


def recode(values: pd.DataFrame, table: pd.DataFrame, how: str, min_coverage: float = 0.99) -> pd.DataFrame:
    """Recode values from source to target codes with a correspondence table.

    Parameters
    ----------
    values : pandas.DataFrame
        Columns ``code`` and ``value`` for one indicator and period.
    table : pandas.DataFrame
        Output of :func:`correspondence`.
    how : {"sum", "wmean", "none"}
        ``sum`` for extensive quantities (sum of ``w_from * value``), ``wmean`` for intensive quantities
        (mean of the source values weighted by ``w_to``), ``none`` keeps 1:1 mappings only.
    min_coverage : float
        A target value is kept only if the source pieces present make up at least this share of the target
        region's population (sum of ``w_to``).

    Returns
    -------
    pandas.DataFrame
        ``code`` (target), ``value``, ``coverage``.
    """
    m = values[["code", "value"]].merge(table, left_on="code", right_on="code_from")
    if how == "none":
        m = m[(m.w_from == 1.0) & (m.w_to == 1.0)]
    if m.empty:
        return pd.DataFrame({"code": [], "value": [], "coverage": []})
    m = m.assign(part=(m.w_from if how == "sum" else m.w_to) * m.value)
    g = m.groupby("code_to").agg(part=("part", "sum"), coverage=("w_to", "sum")).reset_index()
    if how != "sum":
        g["part"] = g.part / g.coverage
    g = g[g.coverage >= min_coverage - 1e-9]
    return pd.DataFrame(
        {"code": g.code_to.values, "value": g.part.values, "coverage": g.coverage.clip(upper=1).values}
    )


def adjacency_pairs(version: str = "2024", cache_dir: str = "data/raw/gisco") -> set[tuple[str, str]]:
    """Pairs of NUTS codes of the same level whose 1:1 million boundaries touch (cached as parquet)."""
    cache = Path(cache_dir) / f"nuts_adjacency_{version}.parquet"
    if cache.exists():
        t = pd.read_parquet(cache)
        return set(zip(t.a, t.b, strict=True))
    import shapely

    src = download(
        GISCO + f"nuts/parquet/NUTS_RG_01M_{version}_3035.parquet",
        Path(cache_dir) / f"NUTS_RG_01M_{version}_3035.parquet",
    )
    raw = pd.read_parquet(src, columns=["NUTS_ID", "LEVL_CODE", "Shape"])
    pairs = []
    for _, grp in raw.groupby("LEVL_CODE"):
        codes = grp.NUTS_ID.to_numpy()
        geoms = shapely.buffer(shapely.from_wkb(grp.Shape.values), 50.0)  # 50 m closes generalisation gaps
        tree = shapely.STRtree(geoms)
        left, right = tree.query(geoms, predicate="intersects")
        pairs += [(codes[i], codes[j]) for i, j in zip(left, right, strict=True) if i < j]
    t = pd.DataFrame(pairs, columns=["a", "b"])
    t.to_parquet(cache)
    return set(pairs)


# ---------------------------------------------------------------------------------------------- schemes
class NutsScheme:
    """NUTS regions of one version as a ``GeoScheme``.

    Parameters
    ----------
    version : str
        ``"2021"`` or ``"2024"``.
    cache_dir : str
        Directory for GISCO and Eurostat files.
    population : mapping of str to float, optional
        Population per NUTS code (any level); edge population weights fall back to area shares where it is
        missing.
    levels : iterable of int
        Levels to include.
    """

    def __init__(
        self,
        version: str = "2024",
        cache_dir: str = "data/raw/gisco",
        population: Mapping[str, float] | None = None,
        levels: Iterable[int] = (0, 1, 2, 3),
    ):
        self.name = f"NUTS{version}"
        self.version = version
        self.cache_dir = cache_dir
        self.levels = tuple(levels)
        self.population = dict(population or {})
        self._table: pd.DataFrame | None = None

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            t = nuts_table(self.version, self.cache_dir)
            self._table = t[t.level.isin(self.levels)].reset_index(drop=True)
        return self._table

    def geo_id(self, code: str) -> str:
        return qualify(self.name, code)

    def units(self, level: int | None = None) -> list[GeoUnit]:
        t = self.table if level is None else self.table[self.table.level == level]
        return [
            GeoUnit(
                geo_id=self.geo_id(r.code),
                scheme=self.name,
                level=int(r.level),
                name=r.name,
                country=r.country,
                area_km2=float(r.area_km2),
                population=_finite(self.population.get(r.code)),
                valid_from=f"{self.version}-01-01",
            )
            for r in t.itertuples()
        ]

    def edges(self) -> list[GeoEdge]:
        t = self.table
        area = dict(zip(t.code, t.area_km2, strict=True))
        codes = set(t.code)
        out = []
        for r in t.itertuples():
            parent = self._parent_code(r.code, codes)
            if parent is None:
                continue
            pc, pp = _finite(self.population.get(r.code)), _finite(self.population.get(parent))
            w_area = area[r.code] / area[parent] if area[parent] > 0 else np.nan
            w_pop = pc / pp if pc is not None and pp else w_area
            out.append(GeoEdge(self.geo_id(r.code), self.geo_id(parent), float(w_pop), float(w_area)))
        return out

    def _parent_code(self, code: str, codes: set[str]) -> str | None:
        for n in range(len(code) - 1, 1, -1):
            if code[:n] in codes:
                return code[:n]
        return None

    def harmonise(self, codes: Iterable[str], source_version: str) -> dict[str, list[tuple[str, float]]]:
        """Map codes of ``source_version`` to this version as ``code -> [(target code, w_from)]``.

        Codes valid in this version map to themselves; codes without a correspondence map to ``[]``.
        """
        valid = set(nuts_table(self.version, self.cache_dir).code)
        table = (
            correspondence(source_version, self.version, self.cache_dir)
            if source_version != self.version
            else pd.DataFrame(columns=["code_from", "code_to", "w_from"])
        )
        groups = {k: list(zip(g.code_to, g.w_from, strict=True)) for k, g in table.groupby("code_from")}
        out = {}
        for c in codes:
            if c in groups:
                out[c] = groups[c]
            elif c in valid:
                out[c] = [(c, 1.0)]
            else:
                out[c] = []
        return out


class LauScheme:
    """Local administrative units of one year, linked to NUTS-3 of the matching NUTS version.

    LAU identifiers are ``LAU{version}:{country}:{lau_code}``; the level is 4 (below NUTS-3).
    """

    def __init__(
        self, version: str = "2024", nuts_version: str | None = None, cache_dir: str = "data/raw/gisco"
    ):
        self.name = f"LAU{version}"
        self.version = version
        self.nuts_scheme = f"NUTS{nuts_version or version}"
        self.cache_dir = cache_dir
        self._table: pd.DataFrame | None = None

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = lau_table(self.version, self.cache_dir)
        return self._table

    def geo_id(self, country: str, lau_code: str) -> str:
        return f"{self.name}:{country}:{lau_code}"

    def units(self, level: int | None = None) -> list[GeoUnit]:
        if level not in (None, LAU_LEVEL):
            return []
        return [
            GeoUnit(
                geo_id=self.geo_id(r.country, r.lau_code),
                scheme=self.name,
                level=LAU_LEVEL,
                name=r.name,
                country=r.country,
                area_km2=_finite(r.area_km2),
                population=_finite(r.population),
                valid_from=f"{self.version}-01-01",
            )
            for r in self.table.itertuples()
        ]

    def edges(self) -> list[GeoEdge]:
        t = self.table
        pop = t.groupby("nuts3").population.transform("sum")
        area = t.groupby("nuts3").area_km2.transform("sum")
        return [
            GeoEdge(
                child=self.geo_id(r.country, r.lau_code),
                parent=qualify(self.nuts_scheme, r.nuts3),
                weight_pop=float(r.population / p) if p > 0 else float(r.area_km2 / a),
                weight_area=float(r.area_km2 / a) if a > 0 else 0.0,
            )
            for r, p, a in zip(t.itertuples(), pop, area, strict=True)
        ]

    def harmonise(self, codes: Iterable[str], source_version: str) -> dict[str, list[tuple[str, float]]]:
        if source_version != self.version:
            raise NotImplementedError("LAU version correspondence")
        return {c: [(c, 1.0)] for c in codes}


def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None
