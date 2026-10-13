import pytest
import yaml

from regsim.registry import IndicatorRegistry
from regsim.schema import Aggregation, DType, Transform


def test_eurostat_registry_loads():
    reg = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    assert len(reg) == 25
    ind = reg.indicator("pop_density")
    assert ind.dtype is DType.CONTINUOUS and ind.transform is Transform.LOG
    assert ind.aggregation is Aggregation.RATIO and ind.ratio_of == ("population", "area_km2")
    assert reg["unemployment_rate"].source.filters["age"] == "Y20-64"
    assert reg.population_weights["dataset"] == "demo_r_pjangrp3"
    assert reg.model_indicators()[0] == "pop_density"


def _write(tmp_path, name, indicators, **top):
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"registry_version": name, **top, "indicators": indicators}))
    return p


def entry(i, connector, **kw):
    return {"indicator_id": i, "dtype": "continuous", "source": {"connector": connector}, **kw}


def test_same_as_and_precedence(tmp_path):
    a = _write(tmp_path, "a.yaml", [entry("gdp", "eurostat"), entry("pop", "eurostat")])
    b = _write(
        tmp_path,
        "b.yaml",
        [
            entry("d36_gdp", "d36", same_as="gdp"),
            entry("d36_heat", "d36"),
            entry("ardeco_gdp", "ardeco", same_as="d36_gdp"),
        ],
    )
    reg = IndicatorRegistry.from_yaml(a, b)
    canon = reg.canonical()
    assert canon["d36_gdp"] == "gdp" and canon["ardeco_gdp"] == "gdp" and canon["d36_heat"] == "d36_heat"
    assert reg.model_indicators() == ["gdp", "pop", "d36_heat"]
    assert reg.redundancy() == [("gdp", ["ardeco_gdp", "d36_gdp"])]
    # Precedence decides which copy enters the model.
    reg2 = IndicatorRegistry(list(reg), precedence=["d36", "eurostat"])
    assert reg2.canonical()["gdp"] == "d36_gdp"


def test_invalid_entries(tmp_path):
    with pytest.raises(ValueError, match="unknown indicator"):
        IndicatorRegistry.from_yaml(_write(tmp_path, "c.yaml", [entry("x", "d36", same_as="nope")]))
    with pytest.raises(ValueError, match="duplicate"):
        IndicatorRegistry.from_yaml(_write(tmp_path, "d.yaml", [entry("x", "d36"), entry("x", "d36")]))
    with pytest.raises(ValueError):
        IndicatorRegistry.from_yaml(_write(tmp_path, "e.yaml", [entry("x", "d36", polarity=2)]))
    with pytest.raises(ValueError):
        IndicatorRegistry.from_yaml(_write(tmp_path, "f.yaml", [entry("x", "d36", dtype="text")]))
