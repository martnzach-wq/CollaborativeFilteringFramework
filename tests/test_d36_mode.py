import sqlite3

import numpy as np
import pandas as pd
import pytest

from regsim.similarity.d36 import D36Scorer, koppen_distance, load_tables

CONFIG = {
    "tier_weights": {"default": {"T1": 0.4, "T2": 0.25, "T3": 0.15, "T4": 0.2}},
    "trf_classes": [["TRF1", 80.0], ["TRF2", 60.0], ["TRF3", 40.0], ["TRF4", 20.0], ["TRF5", -1e9]],
    "categorical": {"koppen": "koppen", "coast": "binary", "degurba": "hierarchical"},
    "empty_tier_distance": 1.0,
    "cics": {
        "urban": {
            "T1": {"temp": 1.0, "elev": 4.0, "koppen": 1.0, "coast": 1.0},
            "T2": {"heat": 1.0, "elev": 4.0},
            "T3": {"degurba": 1.0},
            "T4": {"road": 1.0},
        }
    },
}


@pytest.fixture
def tables(tmp_path):
    rows = []
    vals = {
        "D": {
            "temp": 0.5,
            "elev": 0.2,
            "heat": 0.1,
            "road": 0.5,
            "koppen": "Dfa",
            "coast": "1",
            "degurba": "1",
        },
        "A": {
            "temp": 0.5,
            "elev": 0.2,
            "heat": 0.1,
            "road": 0.5,
            "koppen": "Dfa",
            "coast": "1",
            "degurba": "1",
        },
        "B": {
            "temp": 0.9,
            "elev": 0.6,
            "heat": 0.5,
            "road": 0.1,
            "koppen": "Cfa",
            "coast": "0",
            "degurba": "3",
        },
        "C": {
            "temp": 0.5,
            "elev": 0.2,
            "heat": None,
            "road": 0.5,
            "koppen": "BSk",
            "coast": "1",
            "degurba": None,
        },
    }
    for g, v in vals.items():
        for k, x in v.items():
            if x is None:
                continue
            num, txt = (x, None) if isinstance(x, float) else (None, x)
            rows.append((g, 3, k, num, txt))
    path = tmp_path / "x.gpkg"
    con = sqlite3.connect(path)
    pd.DataFrame(
        rows, columns=["region_geocode", "level", "indicator_id", "normalised_value", "value_text"]
    ).to_sql("knowing.indicator_normalised_values", con, index=False)
    con.close()
    return load_tables(path)


def test_koppen_distance():
    assert koppen_distance("Dfa", "Dfa") == 0.0
    assert koppen_distance("Dfa", "Cfa") == 0.25 and koppen_distance("Dfa", "Dfb") == 0.25
    assert koppen_distance("Dfa", "Dwd") == 0.6 and koppen_distance("Dfa", "BSk") == 1.0


def test_scores(tables):
    s = D36Scorer(tables, CONFIG).score("D", "urban", targets=["A", "B", "C"], contributions=True)
    assert (
        s.loc["A", "composite_distance"] == pytest.approx(0.0)
        and s.loc["A", "transferability_class"] == "TRF1"
    )
    # B: T1 = (0.4 + 4*0.4 + 0.25 + 1) / 7, T2 = (0.4 + 4*0.4) / 5, T3 = 1 (degurba 1 vs 3), T4 = 0.4
    t1, t2 = (0.4 + 1.6 + 0.25 + 1) / 7, 2.0 / 5
    assert s.loc["B", "tier1_distance"] == pytest.approx(t1) and s.loc[
        "B", "tier2_distance"
    ] == pytest.approx(t2)
    comp = 0.4 * t1 + 0.25 * t2 + 0.15 * 1.0 + 0.2 * 0.4
    assert s.loc["B", "composite_distance"] == pytest.approx(comp)
    assert s.loc["B", "composite_score"] == pytest.approx(100 * (1 - comp))
    # C: missing heat uses elevation only in T2; missing degurba makes T3 empty -> distance 1.
    assert s.loc["C", "tier2_distance"] == pytest.approx(0.0) and s.loc["C", "tier3_distance"] == 1.0
    assert s.loc["C", "coverage_ratio"] < 1.0
    assert np.isfinite(s.loc["B", "p:T1:temp"])
