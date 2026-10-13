import pandas as pd
import pytest
import yaml

from regsim.build import build
from regsim.config import RunConfig
from regsim.store.store import ObservationStore


class StubConnector:
    """Returns hand-made Eurostat-like observations; XX012/XX013 are NUTS 2021 codes."""

    def __init__(self):
        self.specs = []

    def fetch(self, spec):
        self.specs.append(spec)
        rows = {
            "demo_r_pjangrp3": [
                ("XX", 200.0),
                ("XX0", 200.0),
                ("XX01", 200.0),
                ("XX014", 100.0),
                ("XX015", 100.0),
                ("YY013", 50.0),
                ("EU27_2020", 1.0),
            ],
            "rate": [("XX012", 2.0), ("XX013", 4.0), ("XX015", 1.0), ("UKC11", 9.0)],
            "count": [("XX014", 10.0), ("XX015", 5.0)],
        }[spec["dataset"]]
        return pd.DataFrame(
            {
                "geo_code": [g for g, _ in rows],
                "indicator_id": spec["indicator_id"],
                "period": "2020",
                "scenario": "observed",
                "value": [v for _, v in rows],
                "component": None,
                "se": float("nan"),
                "source": f"eurostat:{spec['dataset']}",
                "source_version": "test",
                "flags": "",
                "unit": "NR",
            }
        )


@pytest.fixture
def config(cache, tmp_path):
    reg = tmp_path / "reg.yaml"
    reg.write_text(
        yaml.safe_dump(
            {
                "registry_version": "t",
                "indicators": [
                    {
                        "indicator_id": "rate",
                        "dtype": "continuous",
                        "aggregation": "wmean_pop",
                        "source": {"connector": "eurostat", "dataset": "rate"},
                    },
                    {
                        "indicator_id": "count",
                        "dtype": "count",
                        "aggregation": "sum",
                        "source": {"connector": "eurostat", "dataset": "count"},
                    },
                ],
            }
        )
    )
    raw = tmp_path / "raw"
    (raw / "gisco").mkdir(parents=True)
    for f in pd.io.common.Path(cache).glob("*.parquet"):
        (raw / "gisco" / f.name).write_bytes(f.read_bytes())
    return RunConfig.model_validate(
        {
            "data": {
                "indicator_registry": str(reg),
                "levels": [0, 1, 2, 3],
                "periods": ["2018/2022"],
                "store": str(tmp_path / "store"),
                "raw_dir": str(raw),
            }
        }
    )


def test_build_harmonises_and_stores(config):
    stub = StubConnector()
    report = build(config, connector=stub)
    assert report.units_per_level == {0: 2, 1: 2, 2: 2, 3: 4}
    store = ObservationStore(config.data.store)
    o = store.observations().set_index(["geo_id", "indicator_id"])
    # XX012 (rate 2) and XX013 (rate 4) merge into XX014 with LAU population weights 30:20 (plus half of XX011,
    # which is missing, so the recoded value is not kept); XX015 is published natively.
    assert ("NUTS2024:XX014", "rate") not in o.index
    assert o.loc[("NUTS2024:XX015", "rate"), "value"] == pytest.approx(1.0)
    assert o.loc[("NUTS2024:XX014", "count"), "value"] == pytest.approx(10.0)
    assert ("NUTS2024:XX", "population") in o.index
    per = report.to_frame().set_index("indicator_id")
    assert per.loc["rate", "dropped"] == 1 and "UKC11" in per.loc["rate", "dropped_codes"]
    assert per.loc["population", "dropped_codes"] == ["EU27_2020"]
    edges = store.query("SELECT * FROM geo_edge").set_index("child")
    assert edges.loc["NUTS2024:XX014", "weight_pop"] == pytest.approx(0.5)
    assert (pd.io.common.Path(config.data.store) / "build.hash").read_text().strip() == config.config_hash()
    assert all(s["since"] == 2018 and s["until"] == 2022 for s in stub.specs)


def test_free_dimension_is_rejected(config):
    class Mixed(StubConnector):
        def fetch(self, spec):
            f = super().fetch(spec)
            if spec["dataset"] == "rate":
                f = pd.concat([f, f.assign(unit="PC")])
            return f

    with pytest.raises(ValueError, match="free dimensions"):
        build(config, connector=Mixed())
