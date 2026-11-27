"""Strategy interface (decision D57) on a synthetic city configuration."""

import dataclasses

import pandas as pd
import pytest
from pydantic import ValidationError

from regsim.causal.strategy import apply_strategy, check_outcomes
from regsim.config import StrategyFile


@dataclasses.dataclass
class _Mobility:
    parking_cost_base: float = 1.5
    policy_parking_cost_adaptation: dict = dataclasses.field(default_factory=dict)
    policy_capacity_adaptation: dict = dataclasses.field(default_factory=dict)
    policy_area_toll: dict = dataclasses.field(default_factory=dict)
    area_toll_affected_share: float = 1.0
    ev_pi_trajectory: dict = dataclasses.field(default_factory=lambda: {2019: 1.0, 2050: 50.0})


@dataclasses.dataclass
class _Demand:
    overrides: dict = dataclasses.field(default_factory=dict)
    policy_energy: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _City:
    mobility: _Mobility
    energy_demand: _Demand


SRC = {"document": "test strategy"}


def _strategy(**kw):
    base = {"region": "XX", "name": "test", "measures": [], "pathways": [], "outcomes": []}
    return StrategyFile.model_validate(base | kw)


def test_unknown_measure_is_rejected():
    with pytest.raises(ValidationError):
        _strategy(measures=[{"measure": "teleportation", "unit": "x", "intensity": {2030: 1}, "source": SRC}])


def test_measures_pathways_and_report():
    s = _strategy(
        measures=[
            {"measure": "parking_management", "unit": "EUR/trip", "intensity": {2030: 0.3}, "source": SRC},
            {
                "measure": "traffic_calming",
                "unit": "share_removed",
                "intensity": {2030: 0.1, 2040: 0.2},
                "source": SRC,
            },
            {
                "measure": "area_toll",
                "unit": "EUR/entry",
                "intensity": {2030: 5.0},
                "exposure": {"kind": "zone_trip_share", "value": 0.2},
                "source": SRC,
            },
            {"measure": "ev_fleet", "unit": "percent_of_fleet", "intensity": {2040: 60.0}, "source": SRC},
            {"measure": "retrofit", "unit": "percent_per_year", "intensity": {2030: 2.0}, "source": SRC},
            {
                "measure": "energy_households",
                "unit": "share_of_cets",
                "intensity": {2040: 0.5},
                "source": SRC,
            },
            {"measure": "pv", "unit": "MWp", "intensity": {2030: None}, "source": SRC},
        ],
        pathways=[
            {
                "row": "EN_finEN_hh_space heating_fossil fuel",
                "values": {2019: 900.0, 2040: 0.0},
                "source": SRC,
            }
        ],
        outcomes=[
            {"quantity": "emissions_total", "relative_to": 2019, "values": {2040: -0.9}, "source": SRC}
        ],
    )
    city = _City(_Mobility(), _Demand())
    cfg, report = apply_strategy(None, city, s)
    m = cfg.mobility
    assert m.policy_parking_cost_adaptation == {2019: 1.0, 2030: pytest.approx(1.2)}
    assert m.policy_capacity_adaptation == {2019: 1.0, 2030: pytest.approx(0.9), 2040: pytest.approx(0.8)}
    assert m.policy_area_toll == {2019: 0.0, 2030: 5.0} and m.area_toll_affected_share == 0.2
    assert m.ev_pi_trajectory == {2019: 1.0, 2040: 60.0}
    assert cfg.energy_demand.overrides["EN_finEN_hh_space heating_fossil fuel"] == {2019: 900.0, 2040: 0.0}
    assert cfg.energy_demand.policy_energy == {"households": {2019: 0.0, 2040: 0.5}}
    status = report.set_index("element").status
    assert status["retrofit"] == "unmapped" and status["pv"] == "unmapped"
    assert status["emissions_total"] == "check"
    assert city.mobility.policy_parking_cost_adaptation == {}  # the input configuration is unchanged


def test_check_outcomes_relative_target():
    s = _strategy(
        outcomes=[{"quantity": "emissions_total", "relative_to": 2019, "values": {2040: -0.9}, "source": SRC}]
    )
    results = pd.DataFrame({"emissions_total": [1000.0, 150.0]}, index=pd.Index([2019, 2040], name="year"))
    out = check_outcomes(results, s)
    assert out.loc[0, "model"] == pytest.approx(-0.85)
    assert out.loc[0, "gap"] == pytest.approx(0.05)


def test_vienna_template_validates():
    s = StrategyFile.from_yaml("configs/strategies/vienna_klimafahrplan.yaml")
    assert s.region == "AT130" and len(s.measures) >= 5


@dataclasses.dataclass
class _Micro:
    calibration_csv: str | None = "mc.csv"
    city_area_km2: float | None = None
    coupling: str | None = None
    policy_plant_trees: dict = dataclasses.field(default_factory=dict)
    policy_unseal: dict = dataclasses.field(default_factory=dict)
    plant_trees: dict = dataclasses.field(default_factory=dict)
    unseal_ha: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _MicroCity:
    mobility: _Mobility
    energy_demand: _Demand
    microclimate: _Micro


def test_microclimate_levers_in_natural_units():
    s = _strategy(
        measures=[
            {"measure": "greening", "unit": "trees", "intensity": {2030: 5000.0}, "source": SRC},
            {"measure": "unsealing", "unit": "ha", "intensity": {2040: 80.0}, "source": SRC},
        ]
    )
    cfg, report = apply_strategy(None, _MicroCity(_Mobility(), _Demand(), _Micro()), s)
    status = report.set_index("element")
    assert cfg.microclimate.plant_trees == {2019: 0.0, 2030: 5000.0}
    assert status.loc["unsealing", "status"] == "unmapped"  # ha needs the city area
    cfg, report = apply_strategy(None, _MicroCity(_Mobility(), _Demand(), _Micro(city_area_km2=159.2)), s)
    assert cfg.microclimate.unseal_ha == {2019: 0.0, 2040: 80.0}
    assert (report.status == "mapped").all()


def test_check_outcomes_reference_year_outside_run():
    s = _strategy(
        outcomes=[{"quantity": "emissions_total", "relative_to": 2005, "values": {2030: -0.5}, "source": SRC}]
    )
    results = pd.DataFrame({"emissions_total": [1000.0, 500.0]}, index=pd.Index([2019, 2030], name="year"))
    out = check_outcomes(results, s)
    assert out.loc[0, "note"] == "reference year 2005 outside the model run"
