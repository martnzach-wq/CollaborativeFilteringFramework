"""Adapter for the KNOWING integrated SD model (martnzach-wq/KNOWING-integrated-SD-model).

The model is a separate repository; `load` imports it from a local checkout. `runner` returns a
function ``run(scales) -> DataFrame`` for `sd_bridge.perturbation_elasticities`: ``scales`` maps the
names of input groups (`INPUT_GROUPS`) or policy levers to multiplicative factors, applied to a copy
of the city configuration. Input groups follow the perturbation protocol (docs/causal/
sd_model_review.md): consistent groups are scaled together (all mobility population inputs),
constants and series alike (fuel cost as series and as constant).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = np.arange(2019, 2051)

# input group -> MobilityConfig fields scaled together (constants and keyframe series)
INPUT_GROUPS: dict[str, tuple[str, ...]] = {
    "population": (
        "population_internal",
        "population_od",
        "pop_net_increase_internal",
        "pop_net_increase_od",
    ),
    "population (internal)": ("population_internal",),
    "population growth": ("pop_net_increase_internal", "pop_net_increase_od"),
    "cooling degree days": ("cooling_degree_days",),
    "fuel car cost per km": ("policy_car_costs_fuel", "car_cost_per_km_fuel_constant"),
    "EV car cost per km": ("policy_car_costs_ev", "car_cost_per_km_ev_constant"),
    "parking cost (base)": ("parking_cost_base",),
    "value of time": ("value_of_time_base",),
}
SUPPLY_GROUPS: dict[str, tuple[str, ...]] = {"grid emission factor": ("grid_emission_factor",)}
LEVERS = (
    "policy_parking_cost_adaptation",
    "policy_area_toll",
    "policy_road_pricing",
    "policy_cycling_attractiveness",
    "policy_car_passenger_attractiveness",
    "policy_pt_traveltime_adaptation",
    "policy_capacity_adaptation",
    "policy_street_green",
)


@dataclasses.dataclass
class Knowing:
    """Modules of the KNOWING model imported from a checkout."""

    path: Path
    city_config: object
    mobility: object
    energy_demand: object
    energy_supply: object

    def city(self, name: str):
        return getattr(self.city_config, name.upper())


def load(path: str | Path) -> Knowing:
    """Import the KNOWING model from the checkout at ``path``."""
    path = Path(path).resolve()
    if not (path / "city_config.py").exists():
        raise FileNotFoundError(f"no KNOWING model in {path} (city_config.py missing)")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    mods = [importlib.import_module(m) for m in ("city_config", "mobility", "energy_demand", "energy_supply")]
    return Knowing(path, *mods)


def _scale(value, factor: float):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _scale(v, factor) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_scale(v, factor) for v in value)
    if isinstance(value, bool):
        return value
    return value * factor


def configure(city, scales: dict[str, float], calibration: bool | None = None, removed=()):
    """Copy of ``city`` with input groups, supply groups and levers scaled; levers in ``removed`` are held
    at their 2019 value (their neutral value: 0 for additive levers, 1 for factors); ``calibration``
    overrides ``apply_cdd_baseattr_compensation`` (the Tallinn GC-neutral calibration)."""
    mob = copy.deepcopy(city.mobility)
    sup = copy.deepcopy(city.energy_supply)
    for lever in removed:
        series = getattr(mob, lever)
        if series:
            start = series.get(2019, series[min(series)])
            setattr(mob, lever, dict.fromkeys(series, start))
    for name, factor in scales.items():
        if name in INPUT_GROUPS:
            for f in INPUT_GROUPS[name]:
                setattr(mob, f, _scale(getattr(mob, f), factor))
        elif name in SUPPLY_GROUPS:
            for f in SUPPLY_GROUPS[name]:
                setattr(sup, f, _scale(getattr(sup, f), factor))
        elif name in LEVERS:
            setattr(mob, name, _scale(getattr(mob, name), factor))
        else:
            raise KeyError(f"unknown input group or lever {name!r}")
    if calibration is not None:
        mob.apply_cdd_baseattr_compensation = calibration
    return dataclasses.replace(city, mobility=mob, energy_supply=sup)


def runner(
    model: Knowing,
    city: str,
    pi_scenario: float = 0.0,
    calibration: bool | None = None,
    pkm_method: str = "centroid",
):
    """``run(scales) -> DataFrame`` (index year 2019-2050) with car_km_cap, active_km_cap, car_km_share
    (car and car passenger), ev_share, final_energy, elec_demand, emissions_total and emissions_transport.

    ``pkm_method`` "centroid" (length-class centroids; matches the ABM light passenger-km of the
    demonstrators, decision D22) or "mode_specific" (mode-specific trip lengths; used in the M7 runs)."""
    base = model.city(city)
    method = pkm_method

    def run(scales: dict[str, float], removed=()) -> pd.DataFrame:
        c = configure(base, scales, calibration, removed)
        mob = model.mobility.run_mobility(c, pi_scenario=pi_scenario)
        ed = model.energy_demand.run_energy_demand(c, mob, pkm_method=method)
        es = model.energy_supply.run_energy_supply(c, ed)
        m = mob[method] if method in mob else mob["centroid"]
        car = np.asarray(m["dist_car_fossil_km_cap_yr"]) + np.asarray(m["dist_car_electric_km_cap_yr"])
        out = pd.DataFrame(
            {
                "car_km_cap": car,
                "active_km_cap": np.asarray(m["dist_active_km_cap_yr"]),
                "car_km_share": np.asarray(m["modal_split"]["car"])
                + np.asarray(m["modal_split"]["car passenger"]),
                "ev_share": np.asarray(mob["ev_share"]),
                "final_energy": np.asarray(ed["total_final_energy"]),
                "elec_demand": np.asarray(ed["ED_CoreElectricityDemand"]),
                "emissions_total": np.asarray(es["total_emissions_kt"]),
                "emissions_transport": np.asarray(es["em_transport"]),
            },
            index=pd.Index(YEARS[: len(car)], name="year"),
        )
        return out

    return run


def perturbation_protocol(
    model: Knowing,
    city: str,
    drivers: list[str] | None = None,
    rel_step: float = 0.1,
    years=(2030, 2050),
    pkm_method: str = "centroid",
) -> pd.DataFrame:
    """Driver elasticities by the perturbation protocol: BAU, with the city's calibration as configured,
    and, where the city has the GC-neutral calibration, also without it; population both as internal
    only (M7 runs) and as the consistent group of all mobility population inputs.

    Returns one row per city, driver, variant, output and year with the elasticity
    (relative change of the output divided by the relative step).
    """
    from regsim.causal.sd_bridge import perturbation_elasticities

    drivers = drivers or [*INPUT_GROUPS, *SUPPLY_GROUPS]
    variants = [("as configured", None)]
    if model.city(city).mobility.apply_cdd_baseattr_compensation:
        variants.append(("calibration off", False))
    out = []
    for label, cal in variants:
        run = runner(model, city, 0.0, calibration=cal, pkm_method=pkm_method)
        for year in years:
            e = perturbation_elasticities(run, {}, drivers, None, rel_step, year)
            out.append(e.assign(city=city, variant=label, year=year))
    return pd.concat(out, ignore_index=True)


def lever_table(
    model: Knowing, city: str, levers=LEVERS, year: int = 2050, pkm_method: str = "centroid"
) -> pd.DataFrame:
    """Lever effects under full policy (PI = 1): relative change of each output with the lever against
    the lever held at its 2019 value. Levers the city does not configure are left out."""
    from regsim.causal.sd_bridge import lever_effects

    configured = [lv for lv in levers if getattr(model.city(city).mobility, lv)]
    return lever_effects(runner(model, city, 1.0, pkm_method=pkm_method), configured, year).assign(city=city)
