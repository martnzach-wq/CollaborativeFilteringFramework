"""Statistik Austria connector, its build step and the LAU export (no network: files are placed in the cache)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regsim.data.statat import StatAtConnector, _age

DATASET = "OGD_bevstandjbab2002_BevStand"
YEAR = 2026
SUCC = {"62280": ["62252", "62267"]}


def _write_tables(folder: Path) -> None:
    """Tiny OGD table: one municipality, two Vienna districts, one merged municipality, code 0."""
    folder.mkdir(parents=True, exist_ok=True)
    ages = pd.DataFrame(
        {
            "code": ["GALTEJ112-1", "GALTEJ112-66", "GALT5J100-21"],
            "name": ["0 Jahre", "65 Jahre", "100 Jahre und älter"],
        }
    )
    ages.to_csv(folder / f"{DATASET}_{YEAR}_C-GALTEJ112-0.csv", sep=";", index=False)
    counts = {  # municipality: (age 0, age 65, age 100+) per sex
        "10101": (30, 10, 0),
        "90101": (5, 5, 0),
        "90201": (15, 4, 1),
        "62280": (40, 9, 1),
        "0": (7, 0, 0),
    }
    rows = []
    for gem, ns in counts.items():
        for sex in ("C11-1", "C11-2"):
            for code, n in zip(ages.code, ns, strict=True):
                rows.append((f"A10-{YEAR}", sex, f"GRGEMAKT-{gem}", code, n))
    pd.DataFrame(rows, columns=["C-A10-0", "C-C11-0", "C-GRGEMAKT-0", "C-GALTEJ112-0", "F-ISIS-1"]).to_csv(
        folder / f"{DATASET}_{YEAR}.csv", sep=";", index=False
    )


@pytest.fixture
def connector(tmp_path):
    _write_tables(tmp_path / "statat")
    return StatAtConnector(tmp_path / "statat", lau_population={"62252": 1000.0, "62267": 3000.0})


def _spec(measure: str, **kw) -> dict:
    return {"indicator_id": f"x_{measure}", "dataset": DATASET, "year": YEAR, "measure": measure, **kw}


def test_age_labels():
    assert _age("0 Jahre") == 0 and _age("65 Jahre") == 65 and _age("100 Jahre und älter") == 100


def test_table_sums_vienna_and_drops_code_0(connector):
    t = connector.table(DATASET, YEAR)
    assert set(t.lau_code) == {"10101", "90001", "62280"}
    assert t[t.lau_code == "90001"]["count"].sum() == 2 * (5 + 5 + 15 + 4 + 1)


def test_population_split_to_predecessors(connector):
    obs = connector.fetch(_spec("population", successors=SUCC)).set_index("geo_code")
    assert obs.loc["10101", "value"] == 80
    assert obs.loc["90001", "value"] == 60
    assert "62280" not in obs.index
    assert obs.loc["62252", "value"] == pytest.approx(25.0) and obs.loc["62267", "value"] == pytest.approx(
        75.0
    )
    assert obs.loc["62252", "flags"] == "split_from:62280" and obs.loc["10101", "flags"] == ""
    assert set(obs.source) == {"statat"} and set(obs.period) == {"2026"}


def test_share_65plus_copied_to_predecessors(connector):
    obs = connector.fetch(_spec("share_age", min_age=65, successors=SUCC)).set_index("geo_code")
    assert obs.loc["10101", "value"] == pytest.approx(20 / 80)
    assert obs.loc["90001", "value"] == pytest.approx(20 / 60)
    assert obs.loc["62252", "value"] == obs.loc["62267", "value"] == pytest.approx(20 / 100)
    assert obs.loc["62267", "flags"] == "copied_from:62280"


def test_split_without_weights_fails(tmp_path):
    _write_tables(tmp_path / "statat")
    with pytest.raises(ValueError, match="population weights"):
        StatAtConnector(tmp_path / "statat").fetch(_spec("population", successors=SUCC))


def test_build_step_maps_to_lau_ids(connector):
    from regsim.build import BuildReport, _statat
    from regsim.geo.nuts import LauScheme
    from regsim.registry import IndicatorRegistry

    lau = LauScheme("2024")
    lau._table = pd.DataFrame(
        {
            "lau_code": ["10101", "90001", "62252", "62267"],
            "country": "AT",
            "nuts3": "AT111",
            "name": list("abcd"),
            "population": [80.0, 60.0, 1000.0, 3000.0],
            "area_km2": 1.0,
        }
    )
    reg = IndicatorRegistry.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "indicators_statat_v1.yaml"
    )
    report = BuildReport()
    obs = pd.concat(_statat(reg, {"statat": connector}, lau, Path("unused"), report))
    assert set(obs.geo_id) == {"LAU2024:AT:10101", "LAU2024:AT:90001", "LAU2024:AT:62252", "LAU2024:AT:62267"}
    assert set(obs.indicator_id) == {"statat_population", "statat_share_age_65plus"}
    assert all(r["dropped"] == 0 for r in report.per_indicator)
    with pytest.raises(ValueError, match="data.lau"):
        _statat(reg, {"statat": connector}, None, Path("unused"), BuildReport())


def test_export_lau_writes_geopackage(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from regsim.config import DataConfig, RunConfig
    from regsim.export import export_lau
    from regsim.schema import GeoUnit
    from regsim.store.store import ObservationStore

    store = ObservationStore(str(tmp_path / "store"))
    ids = ["10101", "10201"]
    store.write_geo(
        [GeoUnit(f"LAU2024:AT:{c}", "LAU2024", 4, name=c, country="AT", area_km2=1.0) for c in ids]
        + [GeoUnit("LAU2024:SI:001", "LAU2024", 4, name="si", country="SI")],
        [],
    )
    rows = [
        (f"LAU2024:AT:{c}", ind, p, v, "statat", f"x_{p}", fl)
        for c, base in zip(ids, (0.2, 0.3), strict=True)
        for ind, p, v, fl in [
            ("statat_share_age_65plus", "2025", base - 0.1, ""),
            ("statat_share_age_65plus", "2026", base, "copied_from:9" if c == "10201" else ""),
            ("statat_population", "2026", 100.0, ""),
        ]
    ]
    cols = ["geo_id", "indicator_id", "period", "value", "source", "source_version", "flags"]
    store.write(pd.DataFrame(rows, columns=cols))
    regions = gpd.GeoDataFrame(
        {"iso2": ["AT", "AT", "SI"], "level": 4, "code": ["10101", "10201", "001"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
        crs="EPSG:3035",
    )
    gpkg = tmp_path / "d36.gpkg"
    regions.to_file(gpkg, layer="knowing.regions", driver="GPKG")
    cfg = RunConfig(
        data=DataConfig(indicator_registry="unused.yaml", store=str(tmp_path / "store"), d36_gpkg=str(gpkg))
    )

    out, prov = export_lau(
        cfg, tmp_path / "lau_at.gpkg", "AT", ["statat_population", "statat_share_age_65plus"]
    )
    res = gpd.read_file(out, layer="lau")
    assert list(res.lau_code) == ids and res.crs.to_epsg() == 3035
    assert np.allclose(res.statat_share_age_65plus, [0.2, 0.3])  # latest period
    assert res["flags"].fillna("").tolist() == ["", "statat_share_age_65plus:copied_from:9"]
    assert prov["indicators"]["statat_share_age_65plus"]["period"] == "2026" and prov["n_units"] == 2
    assert Path(str(out) + ".provenance.json").exists()
    _, prov25 = export_lau(cfg, tmp_path / "p25.gpkg", "AT", ["statat_share_age_65plus"], period="2025")
    assert prov25["indicators"]["statat_share_age_65plus"]["period"] == "2025"
