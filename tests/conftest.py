import pandas as pd
import pytest
import shapely


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
