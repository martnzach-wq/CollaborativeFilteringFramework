"""Strategy interface (decision D57) on a synthetic city configuration."""

import dataclasses

import pandas as pd
import pytest
from pydantic import ValidationError

from regsim.causal.strategy import apply_strategy, check_outcomes, solve_strategy
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


def _solved_strategy(**kw):
    base = {
        "measures": [
            {
                "measure": "traffic_calming",
                "unit": "share_removed",
                "intensity": {2030: 0.2, 2040: 0.4},
                "role": "solved",
                "solve_group": "shift",
                "confidence": "assumed",
                "source": SRC,
            },
            {"measure": "parking_management", "unit": "factor", "intensity": {2030: 1.2}, "source": SRC},
        ],
        "outcomes": [{"quantity": "trip_share_car", "values": {2030: 0.15}, "source": SRC}],
        "solve": [{"group": "shift", "targets": ["trip_share_car"], "years": [2030]}],
    }
    return _strategy(**(base | kw))


def test_solved_measure_needs_group_and_targets():
    with pytest.raises(ValidationError):
        _strategy(
            measures=[
                {
                    "measure": "traffic_calming",
                    "unit": "share_removed",
                    "intensity": {2030: 0.2},
                    "role": "solved",
                    "source": SRC,
                }
            ]
        )
    with pytest.raises(ValidationError):
        _solved_strategy(solve=[])


def test_solved_measures_are_reported_and_scaled():
    s = _solved_strategy()
    city = _City(_Mobility(), _Demand())
    _, report = apply_strategy(None, city, s)
    rep = report.set_index("element")
    assert rep.loc["traffic_calming", "status"] == "to solve"
    assert rep.loc["traffic_calming", "role"] == "solved"
    assert rep.loc["parking_management", "role"] == "stated"
    cfg, _ = apply_strategy(None, city, s, {"shift": {2030: 0.5}})
    # 2030: 1 - 0.5 x 0.2; 2040 continues with the 2030 scale
    assert cfg.mobility.policy_capacity_adaptation[2030] == pytest.approx(0.9)
    assert cfg.mobility.policy_capacity_adaptation[2040] == pytest.approx(0.8)
    assert cfg.mobility.policy_parking_cost_adaptation[2030] == pytest.approx(1.2)  # stated, unscaled


def test_solve_strategy_meets_the_target():
    s = _solved_strategy()
    city = _City(_Mobility(), _Demand())

    def evaluate(cfg):  # car share falls with the removed capacity: 0.30 - 0.5 x removed share
        removed = 1 - cfg.mobility.policy_capacity_adaptation.get(2030, 1.0)
        return pd.DataFrame({"trip_share_car": [0.30, 0.30 - 0.5 * removed]}, index=pd.Index([2019, 2030]))

    scales, check = solve_strategy(None, city, s, evaluate=evaluate)
    assert scales["shift"][2030] == pytest.approx(1.5, abs=1e-3)  # removed share 0.3
    assert check.loc[0, "gap"] == pytest.approx(0.0, abs=1e-4)


def test_effect_classes_and_registered_measure():
    from regsim.causal import strategy as S

    S.register_measure(
        "test_addon_lever",
        {
            "field": "x_test_lever",
            "units": ("x_guide",),
            "exposure": (),
            "response": "behavioural",
            "status": "lever",
            "effect": "stated effect",
        },
    )
    try:
        s = _strategy(
            measures=[
                {"measure": "test_addon_lever", "unit": "x_guide", "intensity": {2030: 1.0}, "source": SRC},
                {"measure": "ev_fleet", "unit": "percent_of_fleet", "intensity": {2040: 60.0}, "source": SRC},
                {"measure": "parking_management", "unit": "factor", "intensity": {2030: 1.2}, "source": SRC},
            ],
            pathways=[{"row": "EN_finEN_Service fossil", "values": {2040: 0.0}, "source": SRC}],
        )
        cfg, report = apply_strategy(None, _City(_Mobility(), _Demand()), s)
        eff = report.set_index("element").effect
        assert eff["test_addon_lever"] == "stated effect"
        assert eff["ev_fleet"] == "exogenous path"
        assert eff["parking_management"] == "modelled"
        assert eff["EN_finEN_Service fossil"] == "exogenous path"
        assert cfg.mobility.x_test_lever == {2019: 0.0, 2030: 1.0}
        with pytest.raises(ValueError):
            S.register_measure("bad", {"field": "x"})
    finally:
        S.CATALOGUE.pop("test_addon_lever", None)


@dataclasses.dataclass
class _Heat:
    hap0: float = 0.5
    shading_base_rate: float = 0.003
    street_trees: dict = dataclasses.field(default_factory=dict)
    hap_target: dict = dataclasses.field(default_factory=dict)
    shading_rate: dict = dataclasses.field(default_factory=dict)
    ac_growth_mult: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _HeatCity:
    mobility: _Mobility
    energy_demand: _Demand
    heat_health: _Heat | None = None


def test_heat_health_levers_start_from_their_neutral_value():
    s = _strategy(
        measures=[
            {"measure": "street_trees", "unit": "trees_per_year", "intensity": {2026: 4000.0}, "source": SRC},
            {"measure": "heat_action_plan", "unit": "share", "intensity": {2026: 1.0}, "source": SRC},
            {
                "measure": "passive_cooling",
                "unit": "share_per_year",
                "intensity": {2026: 0.025},
                "source": SRC,
            },
            {"measure": "ac_restraint", "unit": "factor", "intensity": {2026: 0.25}, "source": SRC},
        ]
    )
    cfg, report = apply_strategy(None, _HeatCity(_Mobility(), _Demand(), _Heat()), s)
    assert (report.status == "mapped").all()
    assert cfg.heat_health.street_trees == {2019: 0.0, 2026: 4000.0}
    assert cfg.heat_health.hap_target == {2019: 0.5, 2026: 1.0}
    assert cfg.heat_health.shading_rate == {2019: 0.003, 2026: 0.025}
    assert cfg.heat_health.ac_growth_mult == {2019: 1.0, 2026: 0.25}
    assert set(report.effect) == {"modelled", "stated effect"}
    _, report = apply_strategy(None, _HeatCity(_Mobility(), _Demand()), s)
    assert (report.status == "unmapped").all()


def test_backcast_multipliers_and_least_change():
    from regsim.causal.strategy import solve_backcast

    s = _strategy(
        measures=[
            {"measure": "traffic_calming", "unit": "share_removed", "intensity": {2030: 0.2}, "source": SRC},
            {
                "measure": "parking_management",
                "unit": "factor",
                "intensity": {2030: 1.2},
                "backcast_group": "parking",
                "source": SRC,
            },
        ],
        outcomes=[{"quantity": "trip_share_car", "values": {2030: 0.12}, "bound": "max", "source": SRC}],
    )
    city = _City(_Mobility(), _Demand())
    cfg, report = apply_strategy(None, city, s, multipliers={"parking": 0.5, "traffic_calming": 2.0})
    assert cfg.mobility.policy_parking_cost_adaptation[2030] == pytest.approx(1.1)  # 1 + 0.5 x 0.2
    assert cfg.mobility.policy_capacity_adaptation[2030] == pytest.approx(0.6)  # 0.4 removed
    cfg, _ = apply_strategy(None, city, s, multipliers={"parking": 1.0}, delay_to=2030)
    assert cfg.mobility.policy_parking_cost_adaptation[2029] == pytest.approx(1.0)  # delayed

    def evaluate_many(mults):  # car share 0.30 - 0.5 x removed - 0.25 x parking increase
        out = []
        for m in mults:
            c, _ = apply_strategy(None, city, s, multipliers=m)
            removed = 1 - c.mobility.policy_capacity_adaptation[2030]
            park = c.mobility.policy_parking_cost_adaptation[2030] - 1
            out.append(
                pd.DataFrame(
                    {"trip_share_car": [0.30, 0.30 - 0.5 * removed - 0.25 * park]},
                    index=pd.Index([2019, 2030]),
                )
            )
        return out

    groups = {"traffic_calming": (0.0, 2.0), "parking": (0.0, 2.0)}
    sol = solve_backcast(None, city, s, groups, ["trip_share_car"], evaluate_many=evaluate_many)
    assert sol["success"]
    assert sol["margins"].margin.min() == pytest.approx(0.0, abs=1e-4)  # binding target 0.12
    m = sol["multipliers"]  # least change from (1, 1) along (0.1, 0.05): (1.24, 1.12)
    assert m["traffic_calming"] == pytest.approx(1.24, abs=1e-3)
    assert m["parking"] == pytest.approx(1.12, abs=1e-3)
    sol = solve_backcast(
        None, city, s, groups, ["trip_share_car"], objective="effort", evaluate_many=evaluate_many
    )
    assert sol["multipliers"]["traffic_calming"] == pytest.approx(1.8, abs=1e-3)  # cheaper per unit of effort
    assert sol["multipliers"]["parking"] == pytest.approx(0.0, abs=1e-3)
