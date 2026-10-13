import numpy as np
import pandas as pd
import pytest
import shapely

from regsim.geo.nuts import LauScheme, NutsScheme, correspondence, recode
from regsim.geo.scheme import HierarchyGraph, qualify, split_geo_id
from regsim.schema import GeoEdge, GeoUnit


# ------------------------------------------------------------------------------------------ hierarchy graph
def small_graph():
    units = [
        GeoUnit("S:A", "S", 0, population=100.0, area_km2=10.0),
        GeoUnit("S:A1", "S", 1, population=60.0, area_km2=4.0),
        GeoUnit("S:A2", "S", 1, population=40.0, area_km2=6.0),
        GeoUnit("S:x", "S", 2, population=20.0, area_km2=2.0),  # split between A1 and A2
        GeoUnit("S:y", "S", 2, population=50.0, area_km2=2.0),
    ]
    edges = [
        GeoEdge("S:A1", "S:A", 0.6, 0.4),
        GeoEdge("S:A2", "S:A", 0.4, 0.6),
        GeoEdge("S:x", "S:A1", 10 / 60, 0.25),
        GeoEdge("S:x", "S:A2", 10 / 40, 1 / 6),
        GeoEdge("S:y", "S:A1", 50 / 60, 0.5),
    ]
    return HierarchyGraph(units, edges)


def test_graph_navigation():
    g = small_graph()
    assert sorted(p for p, _ in g.parents("S:x")) == ["S:A1", "S:A2"]
    assert g.ancestors("S:x") == ["S:A1", "S:A2", "S:A"]
    assert set(g.descendants("S:A")) == {"S:A1", "S:A2", "S:x", "S:y"}
    assert g.membership("S:x", "S:A1") == pytest.approx(0.5)
    assert g.membership("S:y", "S:A1") == pytest.approx(1.0)
    assert split_geo_id(qualify("NUTS2024", "AT130")) == ("NUTS2024", "AT130")


def test_aggregation_matrix():
    g = small_graph()
    index = ["S:A", "S:A1", "S:A2", "S:x", "S:y"]
    W, parents = g.aggregation_matrix(index, "pop")
    assert parents == [0, 1, 2]
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel()[:3], 1.0)
    pop = np.array([g.units[i].population for i in index])
    M, _ = g.aggregation_matrix(index, "membership")
    # Sums of an extensive quantity reproduce the population of fully covered parents (A, A1).
    assert np.allclose((M @ pop)[:2], pop[:2])
    assert (M @ pop)[2] == pytest.approx(10.0)  # half of x lies in A2
    # Renormalisation over the children present in the index.
    W2, _ = g.aggregation_matrix(["S:A1", "S:y"], "pop")
    assert W2[0, 1] == pytest.approx(1.0)
    assert g.parent_index(index)[3] == [(1, pytest.approx(1 / 6)), (2, pytest.approx(0.25))]


# ------------------------------------------------------------------------------------------ NUTS fixtures
def square(x0, y0, w, h):
    return shapely.to_wkb(shapely.box(x0, y0, x0 + w, y0 + h))


@pytest.fixture
def cache(tmp_path):
    """Tiny GISCO/Eurostat caches.

    Country XX: XX011 -> XX015 (boundary shift, LAUs keep their codes), XX012 + XX013 -> XX014 (merge).
    Country YY: YY011 + YY012 -> YY013, YY014 (re-cut, LAU codes renumbered: polygon fallback).
    """
    changes = pd.DataFrame(
        {
            "code_from": [
                "XX",
                "XX0",
                "XX01",
                "XX011",
                None,
                "XX012",
                "XX013",
                None,
                "YY",
                "YY011",
                "YY012",
                None,
                None,
            ],
            "code_to": [
                "XX",
                "XX0",
                "XX01",
                None,
                "XX015",
                None,
                None,
                "XX014",
                "YY",
                None,
                None,
                "YY013",
                "YY014",
            ],
            "level": [0, 1, 2, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
            "change": [
                None,
                None,
                None,
                "Boundary shift",
                "New region",
                "Merged",
                "Boundary shift",
                "New region",
                None,
                "Boundary shift",
                "Boundary shift",
                "New region",
                "New region",
            ],
        }
    ).astype({"code_from": "string", "code_to": "string", "change": "string"})
    changes.to_parquet(tmp_path / "nuts_changes_2021_2024.parquet")
    lau21 = pd.DataFrame(
        {
            "lau_code": ["1", "2", "3", "4", "a", "b"],
            "country": ["XX", "XX", "XX", "XX", "YY", "YY"],
            "nuts3": ["XX011", "XX011", "XX012", "XX013", "YY011", "YY012"],
            "population": [100.0, 50.0, 30.0, 20.0, 300.0, 100.0],
        }
    )
    lau24 = pd.DataFrame(
        {
            "lau_code": ["1", "2", "3", "4", "c", "d"],
            "country": ["XX", "XX", "XX", "XX", "YY", "YY"],
            "nuts3": ["XX015", "XX014", "XX014", "XX014", "YY013", "YY014"],
            "population": [100.0, 50.0, 30.0, 20.0, 200.0, 200.0],
        }
    )
    for name, t in [("lau_2021", lau21), ("lau_2024", lau24)]:
        t.assign(
            name="", area_km2=1.0, degurba=pd.array([1] * len(t), dtype="Int64"), coastal="no"
        ).to_parquet(tmp_path / f"{name}.parquet")
    # YY011 = left half, YY012 = right half (2021); YY013 = left three quarters, YY014 = right quarter (2024).
    pd.DataFrame(
        {"NUTS_ID": ["YY011", "YY012"], "Shape": [square(0, 0, 2, 2), square(2, 0, 2, 2)]}
    ).to_parquet(tmp_path / "NUTS_RG_01M_2021_3035.parquet")
    pd.DataFrame(
        {"NUTS_ID": ["YY013", "YY014"], "Shape": [square(0, 0, 3, 2), square(3, 0, 1, 2)]}
    ).to_parquet(tmp_path / "NUTS_RG_01M_2024_3035.parquet")
    codes = ["XX", "XX0", "XX01", "XX014", "XX015", "YY", "YY0", "YY01", "YY013", "YY014"]
    pd.DataFrame(
        {
            "code": codes,
            "level": [len(c) - 2 for c in codes],
            "country": [c[:2] for c in codes],
            "name": codes,
            "area_km2": [8.0, 8.0, 8.0, 3.0, 5.0, 8.0, 8.0, 8.0, 6.0, 2.0],
            "mount_type": pd.array([0] * 10, dtype="Int64"),
            "urbn_type": pd.array([0] * 10, dtype="Int64"),
            "coast_type": pd.array([0] * 10, dtype="Int64"),
        }
    ).to_parquet(tmp_path / "nuts_2024.parquet")
    return str(tmp_path)


def test_correspondence_lau_overlap(cache):
    c = correspondence("2021", "2024", cache).set_index(["code_from", "code_to"])
    assert c.loc[("XX012", "XX014"), "w_from"] == pytest.approx(1.0)
    assert c.loc[("XX011", "XX014"), "w_from"] == pytest.approx(50 / 150)
    assert c.loc[("XX011", "XX015"), "w_from"] == pytest.approx(100 / 150)
    assert c.loc[("XX011", "XX014"), "w_to"] == pytest.approx(0.5)
    assert c.loc[("XX011", "XX014"), "method"] == "lau_overlap"
    assert c.loc[("XX", "XX"), "method"] == "identity"


def test_correspondence_polygon_fallback(cache):
    c = correspondence("2021", "2024", cache).set_index(["code_from", "code_to"])
    assert c.loc[("YY011", "YY013"), "w_from"] == pytest.approx(1.0)
    assert c.loc[("YY012", "YY013"), "w_from"] == pytest.approx(0.5)
    assert c.loc[("YY012", "YY014"), "w_from"] == pytest.approx(0.5)
    # Target shares from source populations: YY013 gets 300 from YY011 and 50 from YY012.
    assert c.loc[("YY011", "YY013"), "w_to"] == pytest.approx(300 / 350)
    assert c.loc[("YY012", "YY013"), "method"] == "polygon_overlap"


def test_recode_sum_and_mean(cache):
    c = correspondence("2021", "2024", cache)
    v = pd.DataFrame({"code": ["XX011", "XX012", "XX013"], "value": [150.0, 30.0, 20.0]})
    s = recode(v, c, "sum").set_index("code").value
    assert s["XX014"] == pytest.approx(50 + 30 + 20) and s["XX015"] == pytest.approx(100)
    rates = pd.DataFrame({"code": ["XX011", "XX012", "XX013"], "value": [1.0, 2.0, 4.0]})
    m = recode(rates, c, "wmean").set_index("code").value
    assert m["XX014"] == pytest.approx((50 * 1 + 30 * 2 + 20 * 4) / 100)
    # A missing source piece leaves the target value undefined.
    partial = recode(rates[rates.code != "XX013"], c, "wmean")
    assert "XX014" not in set(partial.code)
    assert set(recode(rates, c, "none").code) == set()


def test_nuts_scheme_units_edges_and_harmonise(cache):
    pop = {"XX": 200.0, "XX0": 200.0, "XX01": 200.0, "XX014": 100.0, "XX015": 100.0}
    s = NutsScheme("2024", cache, population=pop)
    units = {u.geo_id: u for u in s.units()}
    assert units["NUTS2024:XX014"].level == 3 and units["NUTS2024:XX014"].population == 100.0
    assert units["NUTS2024:YY013"].population is None
    edges = {(e.child, e.parent): e for e in s.edges()}
    assert edges[("NUTS2024:XX014", "NUTS2024:XX01")].weight_pop == pytest.approx(0.5)
    # No population for YY: area share is used.
    assert edges[("NUTS2024:YY013", "NUTS2024:YY01")].weight_pop == pytest.approx(6 / 8)
    h = s.harmonise(["XX014", "XX012", "ZZ999"], "2021")
    assert (
        h["XX014"] == [("XX014", 1.0)] and h["XX012"] == [("XX014", pytest.approx(1.0))] and h["ZZ999"] == []
    )
    g = HierarchyGraph(s.units(), s.edges())
    assert g.ancestors("NUTS2024:XX014") == ["NUTS2024:XX01", "NUTS2024:XX0", "NUTS2024:XX"]


def test_lau_scheme(cache):
    lau = LauScheme("2024", cache_dir=cache)
    units = lau.units()
    assert len(units) == 6 and all(u.level == 4 for u in units)
    e = {x.child: x for x in lau.edges()}
    assert e["LAU2024:XX:2"].parent == "NUTS2024:XX014"
    assert e["LAU2024:XX:2"].weight_pop == pytest.approx(0.5)
