"""Transfer indicators (D65) and lever-specific donor features (D56 extension); no network."""

import pandas as pd
import pytest

from regsim.causal import sd_knowing as K
from regsim.registry import IndicatorRegistry


def test_transfer_registry_loads_with_the_eurostat_registry():
    r = IndicatorRegistry.from_yaml(
        "configs/indicators_eurostat_v1.yaml", "configs/indicators_transfer_v1.yaml"
    )
    for i in ("share_age_65plus", "share_age_75plus", "employed_work_other_region", "private_households"):
        assert i in r.entries
        assert "transfer" in r.entries[i].tags


def test_commuter_features_change_the_donor_weights(monkeypatch):
    profiles = {
        "target": {"od_car_trip_share": 0.7, "od_population_ratio": 0.3},
        "a": {"od_car_trip_share": 0.7, "od_population_ratio": 0.3},
        "b": {"od_car_trip_share": 0.3, "od_population_ratio": 1.5},
    }
    base = dict.fromkeys(K.PROFILE_FEATURES, 0.3) | {"polycentric": 0.0}

    def profile(model, city):
        name = city if isinstance(city, str) else city.name
        return pd.Series(base | profiles[name])

    monkeypatch.setattr(K, "mobility_profile", profile)
    w0 = K.donor_weights(None, "target", ["a", "b"])
    assert w0["a"] == pytest.approx(0.5)  # same mobility profile: equal weights
    w1 = K.donor_weights(None, "target", ["a", "b"], features=K.PROFILE_FEATURES + K.COMMUTER_FEATURES)
    assert w1["a"] > 0.6  # the donor with the target's commuter structure counts more
    assert set(K.LEVER_FEATURES) == {"area toll", "road pricing", "parking management"}
