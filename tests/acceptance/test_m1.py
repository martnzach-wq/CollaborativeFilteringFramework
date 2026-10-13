"""M1 acceptance (HANDOVER.md §3, adapted to NUTS 2024 on 2026-10-13, see docs/decisions.md).

Needs a completed `regsim build configs/example.yaml` (store in data/store, raw caches in data/raw).
"""

from pathlib import Path

import pandas as pd
import pytest

from regsim.data.eurostat import EurostatConnector, EurostatError
from regsim.registry import IndicatorRegistry
from regsim.schema import Lens
from regsim.store.store import ObservationStore

STORE = Path("data/store")
GPKG = Path("data/raw/d36/knowing-transferability-export-v2.5.gpkg")
REGISTRY = "configs/indicators_eurostat_v1.yaml"
PROTOTYPE_CELLS = 18_863
WINDOW = "2018/2022"

needs_build = pytest.mark.skipif(not (STORE / "observations").exists(), reason="run `regsim build` first")


def uk_cells(registry: IndicatorRegistry) -> dict[str, int]:
    """UK cells at NUTS-0/2/3 in 2018-2022 per indicator, from the cached Eurostat responses."""
    c = EurostatConnector(cache_dir="data/raw/eurostat", offline=True)
    out = {}
    for e in registry:
        spec = {
            "dataset": e.source.dataset,
            "indicator_id": e.indicator_id,
            "filters": e.source.filters,
            "since": 2013,
            "until": 2022,
            "validate": False,
        }
        try:
            o = c.fetch(spec)
        except EurostatError:
            pytest.skip("Eurostat cache incomplete")
        o = o[o.geo_code.str.startswith("UK") & o.period.astype(int).between(2018, 2022)]
        out[e.indicator_id] = o[o.geo_code.str.len().isin([2, 4, 5])].geo_code.nunique()
    return out


@needs_build
def test_m1_prototype_matrix_without_uk():
    """Observed cells within ±2 % of the prototype minus the UK; indicators without UK data within ±5 %."""
    registry = IndicatorRegistry.from_yaml(REGISTRY)
    store = ObservationStore(str(STORE))
    inds = tuple(e.indicator_id for e in registry)
    m = store.matrix(Lens(indicators=inds), index_levels=[0, 2, 3], period=WINDOW)
    pop = store.matrix(Lens(indicators=("population",)), index_levels=[0, 2, 3], period=WINDOW)
    has_pop = pop.observed[:, 0]
    ours = pd.Series(m.observed[has_pop].sum(axis=0), index=m.indicator_ids)
    uk = pd.Series(uk_cells(registry))
    proto = pd.read_csv("reference/prototype/prototype_indicators.csv").set_index("indicator").n_observed
    base = proto - uk
    total = (ours.sum() - (PROTOTYPE_CELLS - uk.sum())) / (PROTOTYPE_CELLS - uk.sum())
    assert abs(total) <= 0.02, f"observed cells deviate by {100 * total:.2f} %"
    no_uk = uk[uk == 0].index
    rel = (ours[no_uk] - base[no_uk]) / base[no_uk]
    assert (rel.abs() <= 0.05).all(), rel[rel.abs() > 0.05].to_dict()
    print(f"units with population: {int(has_pop.sum())}; cells {int(ours.sum())} vs {int(base.sum())}")


@pytest.mark.skipif(not GPKG.exists(), reason="D3.6 GeoPackage not downloaded")
def test_m1_d36_round_trip():
    """Every row of indicator_values is stored, empty, or outside NUTS/LAU 2024 (UK)."""
    from regsim.data.d36 import read_d36
    from regsim.geo.nuts import nuts_table

    d = read_d36(GPKG, with_scores=False)
    obs, counts = d.observations(valid_nuts=set(nuts_table("2024", "data/raw/gisco").code))
    assert counts["rows"] == 4_117_466
    assert counts["stored"] + counts["empty"] + counts["unmapped"] - counts["array"] == counts["rows"]
    unmapped_regions = set(d.values.region_geocode) - set(d.regions.geocode)
    assert not unmapped_regions
