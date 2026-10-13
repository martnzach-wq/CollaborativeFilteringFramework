import numpy as np
import pandas as pd
import pytest

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
