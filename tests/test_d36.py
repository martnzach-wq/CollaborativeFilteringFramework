import sqlite3

import pandas as pd
import pytest

from regsim.data.d36 import read_d36
from regsim.schema import Aggregation, DType, Transform


@pytest.fixture
def gpkg(tmp_path):
    path = tmp_path / "knowing-transferability-export-v2.5.gpkg"
    con = sqlite3.connect(path)
    pd.DataFrame(
        {
            "fid": [1, 2, 3, 4, 5],
            "id": ["heat_days_1990_2020_mean", "degurba", "population_density", "elevation_max", "SOEC01"],
            "name": ["Heat days", "DEGURBA", "Population density", "Elevation max", "GDP"],
            "units": ["days", "category", "p/km2", "m", "EUR"],
            "data_type": ["continuous", "categorical", "continuous", "continuous", "continuous"],
            "theme": ["Climate", "Typology", "Socioeconomy", "Geography", "Socioeconomy"],
            "subtheme": None,
            "normalisation": ["minmax", "none", "log_minmax", "minmax", "log_minmax"],
            "distance_func": ["manhattan", "hierarchical_binary", "manhattan", "manhattan", "manhattan"],
            "log_transform": [0, 0, 1, 1, 1],
            "source_dataset": None,
            "active": 1,
            "notes": None,
        }
    ).to_sql("knowing.indicator", con, index=False)
    regions = ["AT@NUTS:0:AT", "AT@NUTS:3:AT130", "UK@NUTS:3:UKC11", "AT@LAU:4:90001"]
    rows = []
    for g in regions:
        rows.append((g, "heat_days_1990_2020_mean", "1990-01-01", "2020-12-31", "baseline", 10.0, None, None))
        rows.append(
            (g, "degurba", "2026-01-01", "2026-12-31", "baseline", None, "1" if "130" in g else "3", None)
        )
        rows.append((g, "population_density", "2026-01-01", "2026-12-31", "baseline", None, None, None))
    pd.DataFrame(
        rows,
        columns=[
            "region_geocode",
            "indicator_id",
            "period_start",
            "period_end",
            "rcp",
            "value_numeric",
            "value_text",
            "value_array",
        ],
    ).to_sql("knowing.indicator_values", con, index=False)
    pd.DataFrame(
        {
            "geocode": regions,
            "iso2": ["AT", "AT", "UK", "AT"],
            "code": ["AT", "AT130", "UKC11", "90001"],
            "level": [0, 3, 3, 4],
            "name_latin": regions,
            "geocode_parent": [None, "AT@NUTS:2:AT13", "UK@NUTS:2:UKC1", "AT@NUTS:3:AT130"],
            "mountain_type": None,
            "urban_type": None,
            "coastal_type": None,
            "degurba": None,
            "area": None,
        }
    ).to_sql("knowing.regions", con, index=False)
    pd.DataFrame(
        {
            "demonstrator_geocode": ["AT@NUTS:3:AT130"],
            "target_geocode": ["AT@NUTS:0:AT"],
            "composite_score": [80.0],
        }
    ).to_sql("knowing.precomputed_scores", con, index=False)
    con.close()
    return path


def test_read_and_round_trip(gpkg):
    d = read_d36(gpkg)
    assert d.version == "2.5" and d.n_value_rows == 12 and len(d.scores) == 1
    obs, counts = d.observations(valid_nuts={"AT", "AT130"})
    # 12 rows = 4 empty (population_density) + 2 unmapped (UK) + 6 stored
    assert counts == {"rows": 12, "stored": 6, "empty": 4, "unmapped": 2, "array": 0}
    assert counts["stored"] + counts["empty"] + counts["unmapped"] == counts["rows"]
    o = obs.set_index(["geo_id", "indicator_id"])
    assert o.loc[("NUTS2024:AT130", "d36_heat_days_1990_2020_mean"), "period"] == "1990/2020"
    assert o.loc[("LAU2024:AT:90001", "d36_degurba"), "period"] == "2026"
    assert set(obs.scenario) == {"observed"}  # D3.6 baseline = present state
    # Categorical values are integer codes into the sorted categories ["1", "3"].
    assert o.loc[("NUTS2024:AT130", "d36_degurba"), "value"] == 0.0
    assert o.loc[("NUTS2024:AT", "d36_degurba"), "value"] == 1.0
    assert (obs.source == "d36").all() and (obs.source_version == "2.5").all()


def test_registry_entries(gpkg):
    entries = {e.indicator_id: e for e in read_d36(gpkg).registry_entries()}
    # Unpopulated indicators (SOEC01) are not registered.
    assert set(entries) == {"d36_heat_days_1990_2020_mean", "d36_degurba", "d36_population_density"}
    heat = entries["d36_heat_days_1990_2020_mean"]
    assert heat.aggregation is Aggregation.WMEAN_AREA and heat.temporal == "static"
    assert entries["d36_degurba"].dtype is DType.CATEGORICAL and entries["d36_degurba"].categories == [
        "1",
        "3",
    ]
    dens = entries["d36_population_density"]
    assert dens.same_as == "pop_density" and dens.transform is Transform.LOG1P
