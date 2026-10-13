import numpy as np
import pandas as pd
import pytest

from regsim.schema import GeoEdge, GeoUnit, Lens
from regsim.store.store import ObservationStore
from regsim.synthetic import generate_hierarchical


def obs(rows):
    return pd.DataFrame(rows, columns=["geo_id", "indicator_id", "period", "value", "source"])


@pytest.fixture
def store(tmp_path):
    s = ObservationStore(str(tmp_path))
    units = [GeoUnit("N:A", "N", 0), GeoUnit("N:A1", "N", 1), GeoUnit("N:A2", "N", 1)]
    s.write_geo(units, [GeoEdge("N:A1", "N:A", 0.5, 0.5), GeoEdge("N:A2", "N:A", 0.5, 0.5)])
    s.write(
        obs(
            [
                ("N:A1", "gdp", "2018", 10.0, "eurostat:x"),
                ("N:A1", "gdp", "2019", 20.0, "eurostat:x"),
                ("N:A1", "gdp", "2013", 99.0, "eurostat:x"),
                ("N:A2", "gdp", "2019", 30.0, "eurostat:x"),
                ("N:A", "heat", "2018/2022", 5.0, "eurostat:y"),
            ]
        )
    )
    s.write(
        obs(
            [
                ("N:A1", "d36_gdp", "2020", 1000.0, "d36"),
                ("N:A", "d36_gdp", "2020", 7.0, "d36"),
            ]
        )
    )
    return s


def test_write_and_query(store):
    o = store.observations()
    assert len(o) == 7 and set(o.source) == {"eurostat:x", "eurostat:y", "d36"}
    assert (o.scenario == "observed").all()
    n = store.query("SELECT count(*) AS n FROM geo_edge").n.iloc[0]
    assert n == 2
    # Rewriting a source replaces its file.
    store.write(obs([("N:A2", "d36_gdp", "2020", 1.0, "d36")]))
    assert len(store.query("SELECT * FROM observations WHERE source = 'd36'")) == 1


def test_matrix_window_mean(store):
    m = store.matrix(period="2018/2022")
    f = m.to_frame()
    assert f.loc["N:A1", "gdp"] == pytest.approx(15.0)  # 2013 is outside the window
    assert f.loc["N:A", "heat"] == pytest.approx(5.0)  # window-labelled value
    assert m.geo_ids == ["N:A", "N:A1", "N:A2"] and m.level.tolist() == [0, 1, 1]
    assert m.observed.sum() == 5


def test_matrix_canonical_and_precedence(store):
    canon = {"d36_gdp": "gdp"}
    m = store.matrix(
        Lens(indicators=("gdp",)), period="2018/2022", canonical=canon, precedence=["eurostat", "d36"]
    )
    f = m.to_frame()
    assert f.loc["N:A1", "gdp"] == pytest.approx(15.0) and m.sources[1, 0] == "eurostat:x"
    assert f.loc["N:A", "gdp"] == pytest.approx(7.0)  # only the D3.6 copy exists here
    m2 = store.matrix(Lens(indicators=("gdp",)), period="2018/2022", canonical=canon, precedence=["d36"])
    assert m2.to_frame().loc["N:A1", "gdp"] == pytest.approx(1000.0)


def test_matrix_levels(store):
    m = store.matrix(index_levels=[1], period="2018/2022")
    assert m.geo_ids == ["N:A1", "N:A2"]


def test_write_validates(tmp_path):
    s = ObservationStore(str(tmp_path))
    with pytest.raises(ValueError, match="lack columns"):
        s.write(pd.DataFrame({"geo_id": ["a"]}))
    with pytest.raises(ValueError, match="must not be missing"):
        s.write(obs([(None, "x", "2020", 1.0, "s")]))


def test_synthetic_roundtrip(tmp_path):
    d = generate_hierarchical(
        n_per_level=(3, 9), n_indicators=4, rank=2, sigma_level=(1.0, 0.5), missingness="mcar", seed=1
    )
    s = ObservationStore(str(tmp_path))
    s.write_geo(d.units(), d.edges())
    s.write(d.to_observations())
    m = s.matrix(geo_ids=d.geo_ids, period="P0")
    assert np.array_equal(m.observed, d.observed[0])
    assert np.allclose(m.X[m.observed], d.X[0][d.observed[0]])
