"""Local strategies as inputs of the SD model (decision D57).

A strategy (`regsim.config.StrategyFile`) supplies the intensity and exposure of measures and the trajectories of
bundled sectors; the SD model supplies the responses. `apply_strategy` maps the measures onto the levers of a city
configuration and the pathways onto its MAED-City rows, and lists what it cannot map; `check_outcomes` compares the
model with the outcome targets of the strategy. See `docs/causal/transfer_framework.md`.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd

from regsim.config import StrategyFile

# units that are shares of a physical quantity (capacity, travel time, cycling resistance) and their upper limit
# for solved measures (decision D61)
SHARE_UNITS = ("share_removed", "share_time_saved", "share")
MAX_SHARE = 0.9

# measure -> SD field, accepted units, exposure kinds, response kind, status
CATALOGUE: dict[str, dict] = {
    # energy levers (D58): share of the way from the reference (BAU) to the CETS trajectory, per year
    **{
        f"energy_{g}": {
            "field": ("energy_demand", "policy_energy", g),
            "units": ("share_of_cets",),
            "exposure": (),
            "response": "technical",
            "status": "lever",
            "effect": "exogenous path",
        }
        for g in ("households", "services", "industry", "freight", "transport_other")
    },
    **{
        f"supply_{g}": {
            "field": ("energy_supply", "policy_supply", g),
            "units": ("share_of_cets",),
            "exposure": (),
            "response": "physical",
            "status": "lever",
            "effect": "exogenous path",
        }
        for g in ("pv", "wind", "hydro", "district_heating", "electrolysis", "other")
    },
    "road_pricing": {
        "field": "policy_road_pricing",
        "units": ("EUR/km",),
        "exposure": ("charged_km_per_car_trip",),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "area_toll": {
        "field": "policy_area_toll",
        "units": ("EUR/entry",),
        "exposure": ("zone_trip_share",),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "parking_management": {
        "field": "policy_parking_cost_adaptation",
        "units": ("factor", "EUR/trip"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "traffic_calming": {
        "field": "policy_capacity_adaptation",
        "units": ("factor", "share_removed"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "pt_optimisation": {
        "field": "policy_pt_traveltime_adaptation",
        "units": ("factor", "share_time_saved"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "cycling_optimisation": {
        "field": "cycling_resistance_reduction",
        "units": ("share",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "fifteen_minute_city": {
        "field": "policy_distance_adaptation",
        "units": ("index",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "car_occupancy": {
        "field": "policy_car_passenger_attractiveness",
        "units": ("index",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "ev_fleet": {
        "field": "ev_pi_trajectory",
        "units": ("percent_of_fleet",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "exogenous path",
    },
    # microclimate levers (D60): share of the PALM-4U pathway per year, or natural units placed on the land-cover
    # classes in the proportions of the pathway; decoupled from traffic calming unless the city sets a coupling
    "greening": {
        "field": ("microclimate", {"share_of_pathway": "policy_plant_trees", "trees": "plant_trees"}),
        "units": ("share_of_pathway", "trees"),
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    # heat-health levers (submodel heat_health of the SD model): natural units per year on the land-cover
    # classes (trees, unsealing) or on the dwelling stock; "neutral" is the value without the measure (a number
    # or an attribute of the city's heat_health configuration)
    "street_trees": {
        "field": ("heat_health", {"trees_per_year": "street_trees"}),
        "units": ("trees_per_year",),
        "neutral": 0.0,
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    "courtyard_trees": {
        "field": ("heat_health", {"trees_per_year": "courtyard_trees"}),
        "units": ("trees_per_year",),
        "neutral": 0.0,
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    "unsealing_streets": {
        "field": ("heat_health", {"pp_per_year": "unseal_street_pp"}),
        "units": ("pp_per_year",),
        "neutral": 0.0,
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    "unsealing_courtyards": {
        "field": ("heat_health", {"pp_per_year": "unseal_dense_pp"}),
        "units": ("pp_per_year",),
        "neutral": 0.0,
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    "ac_restraint": {
        "field": ("heat_health", {"factor": "ac_growth_mult"}),
        "units": ("factor",),
        "neutral": 1.0,
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
        "effect": "modelled",
    },
    "heat_action_plan": {
        "field": ("heat_health", {"share": "hap_target"}),
        "units": ("share",),
        "neutral": "hap0",
        "exposure": (),
        "response": "health",
        "status": "lever",
        "effect": "stated effect",
    },
    "care_facility_cooling": {
        "field": ("heat_health", {"share": "care_cooled"}),
        "units": ("share",),
        "neutral": "care_cooled0",
        "exposure": (),
        "response": "health",
        "status": "lever",
        "effect": "stated effect",
    },
    "cool_zones": {
        "field": ("heat_health", {"places": "cool_zone_places"}),
        "units": ("places",),
        "neutral": "cool_zone_places0",
        "exposure": (),
        "response": "health",
        "status": "lever",
        "effect": "stated effect",
    },
    "passive_cooling": {
        "field": ("heat_health", {"share_per_year": "shading_rate"}),
        "units": ("share_per_year",),
        "neutral": "shading_base_rate",
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "stated effect",
    },
    "building_greening": {
        "field": ("heat_health", {"share_per_year": "greening_rate"}),
        "units": ("share_per_year",),
        "neutral": 0.0,
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "stated effect",
    },
    "district_cooling": {
        "field": ("heat_health", {"share": "dc_share"}),
        "units": ("share",),
        "neutral": 0.0,
        "exposure": (),
        "response": "technical",
        "status": "lever",
        "effect": "stated effect",
    },
    "unsealing": {
        "field": ("microclimate", {"share_of_pathway": "policy_unseal", "ha": "unseal_ha"}),
        "units": ("share_of_pathway", "ha"),
        "exposure": (),
        "response": "physical",
        "status": "lever",
        "effect": "modelled",
    },
    "retrofit": {
        "field": None,
        "units": ("percent_per_year",),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
        "effect": "not mapped",
    },
    "heating_change": {
        "field": None,
        "units": ("units_per_year", "share"),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
        "effect": "not mapped",
    },
    "pv": {
        "field": None,
        "units": ("MWp",),
        "exposure": (),
        "response": "physical",
        "status": "bundled",
        "effect": "not mapped",
    },
    "district_heating": {
        "field": None,
        "units": ("share", "GWh"),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
        "effect": "not mapped",
    },
    "electrolysis": {
        "field": None,
        "units": ("MW",),
        "exposure": (),
        "response": "systemic",
        "status": "bundled",
        "effect": "not mapped",
    },
}


# effect classes: "modelled" (the model estimates the response to the intensity), "stated effect" (the size of the
# response comes from an external estimate and is imposed), "exogenous path" (the trajectory of the quantity itself
# is an input), "not mapped" (no lever; give a pathway instead)
EFFECTS = ("modelled", "stated effect", "exogenous path", "not mapped")


def register_measure(name: str, spec: dict) -> None:
    """Add a measure of a city add-on to the catalogue (e.g. a Vienna-specific lever). ``spec`` has the keys of
    the catalogue entries; ``field`` names an attribute of the city's mobility configuration."""
    missing = {"field", "units", "exposure", "response", "status", "effect"} - set(spec)
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")
    if spec["effect"] not in EFFECTS:
        raise ValueError(f"{name}: effect must be one of {EFFECTS}")
    CATALOGUE[name] = dict(spec)


def _series(values: dict[int, float | None], base_year: int, base_value: float) -> dict[int, float]:
    out = {base_year: base_value}
    out.update({y: float(v) for y, v in values.items() if v is not None})
    return dict(sorted(out.items()))


def _convert(measure: str, unit: str, value: float, mobility) -> float:
    if measure == "parking_management" and unit == "EUR/trip":
        return 1 + value / mobility.parking_cost_base
    if unit in ("share_removed", "share_time_saved"):
        return 1 - value
    return value


def _scale_at(scale: dict[int, float], year: int, base_year: int) -> float:
    pts = sorted({base_year: 0.0, **scale}.items())
    return float(np.interp(year, [a for a, _ in pts], [b for _, b in pts]))


def _interp(points: dict, year: int) -> float:
    ys = sorted(points)
    return float(np.interp(year, ys, [float(points[y]) for y in ys]))


def _reference_value(measure, cat: dict, year: int, base, heat) -> float:
    """Value of a measure's lever without the measure, in the measure's unit (backcast multiplier m = 0)."""
    field = cat["field"]
    if "reference" in cat:  # a city add-on gives the reference path of an exogenous-path measure
        return _interp(cat["reference"], year)
    if isinstance(field, tuple) and field[0] == "heat_health":
        n = cat.get("neutral", 0.0)
        return float(getattr(heat, n)) if isinstance(n, str) and heat is not None else float(n or 0.0)
    if measure.measure == "ev_fleet":
        current = getattr(base.mobility, field) or {}
        return _interp(current, year) if current else 0.0
    if isinstance(field, tuple):
        return 0.0
    return 1.0 if measure.unit == "factor" else 0.0


def _upper_limit(measure, cat: dict) -> float:
    field = cat["field"]
    if measure.unit == "percent_of_fleet":
        return 100.0
    if isinstance(field, tuple) and field[0] == "heat_health" and measure.unit == "share":
        return 1.0
    if measure.unit in SHARE_UNITS:
        return MAX_SHARE
    return float("inf")


def _multiply(
    given: dict,
    mult: float,
    ref,
    by: int,
    start: int | None,
    cap: float,
    accel_from: int | None = None,
    ramp: int = 4,
) -> dict:
    """Stated values moved to ref + mult x (stated - ref); with ``accel_from`` a multiplier above 1 instead runs the
    stated path faster from that year (the value of year y is the stated value of accel_from + (y - accel_from) x
    mult), for paths towards an end state; with ``start`` the measure keeps its reference value before ``start`` and
    reaches the multiplied path over ``ramp`` years from ``start`` on."""
    if accel_from is not None and mult > 1.0:
        pts = {by: ref(by), **given}
        last = max(given)
        out = {y: v for y, v in given.items() if y < accel_from}
        out.update(
            {
                y: min(max(_interp(pts, accel_from + (y - accel_from) * mult), 0.0), cap)
                for y in range(accel_from, last + 1)
            }
        )
    else:
        out = {y: min(max(ref(y) + mult * (v - ref(y)), 0.0), cap) for y, v in given.items()}
    if start is not None and start > by:
        # held at the reference before ``start``, then a linear transition over ``ramp`` years to the path [A]
        pts = {by: ref(by), **out}
        last = max(max(out), start + ramp - 1)
        path = {y: _interp(pts, y) for y in range(start, last + 1)}
        out = {y: (ref(y) if by < y < start else v) for y, v in out.items()}
        out[start - 1] = ref(start - 1)
        for y in range(start, last + 1):
            w = min((y - start + 1) / ramp, 1.0)
            out[y] = min(max(ref(y) + w * (path[y] - ref(y)), 0.0), cap)
    return dict(sorted(out.items()))


def apply_strategy(
    model,
    city,
    strategy: StrategyFile,
    scales: dict[str, dict[int, float]] | None = None,
    multipliers: dict[str, float] | None = None,
    delay_to: int | None = None,
    reference_rows: dict[str, dict[int, float]] | None = None,
    accelerate: dict[str, int] | None = None,
) -> tuple[object, pd.DataFrame]:
    """City configuration with the strategy's measures on the levers and its pathways on the MAED-City rows.

    Stated measures enter with their intensity. Solved measures (decision D61) enter with their shape scaled
    by the common scale of their group, ``scales[group][year]`` (linear between the years, 0 in the base year),
    as found by `solve_strategy`; without a scale they are reported as "to solve" and left out.
    Backcast (decision D64): ``multipliers[group]`` moves every element of a backcast group (``backcast_group``,
    default the measure or row) from its reference to reference + m x (stated - reference), with physical caps
    (shares, fleet share, non-negative rows); ``delay_to`` delays the multiplied groups to that year. The reference of a
    pathway row is the city's reference override or ``reference_rows[row]``. For groups in ``accelerate`` (group ->
    year) a multiplier above 1 runs the stated path faster from that year (an earlier end state).
    Returns the configuration and a table of every strategy element with its role and mapping status; elements
    without a lever, without values, or with an unsupported unit or exposure are listed there and left out."""
    from regsim.causal import sd_knowing as K

    base = model.city(city) if isinstance(city, str) else city
    mob = copy.deepcopy(base.mobility)
    demand = copy.deepcopy(base.energy_demand)
    supply = copy.deepcopy(getattr(base, "energy_supply", None))
    micro = copy.deepcopy(getattr(base, "microclimate", None))
    heat = copy.deepcopy(getattr(base, "heat_health", None))
    report = []
    by = strategy.base_year
    for m in strategy.measures:
        cat = CATALOGUE[m.measure]
        row = {
            "element": m.measure,
            "kind": "measure",
            "role": m.role,
            "effect": cat["effect"],
            "source": m.source.document,
        }
        given = {y: v for y, v in m.intensity.items() if v is not None}
        if m.role == "solved" and given:
            if scales is None or m.solve_group not in scales:
                report.append(row | {"status": "to solve", "reason": f"group {m.solve_group}"})
                continue
            raw_neutral = 1.0 if m.unit == "factor" else 0.0
            given = {
                y: raw_neutral + _scale_at(scales[m.solve_group], y, by) * (v - raw_neutral)
                for y, v in given.items()
            }
            if m.unit in SHARE_UNITS:  # a scaled share stays below its physical limit
                given = {y: min(v, MAX_SHARE) for y, v in given.items()}
        group = m.backcast_group or m.measure
        if multipliers is not None and group in multipliers and given and cat["status"] == "lever":
            row["multiplier"] = multipliers[group]

            def ref(y, m=m, cat=cat):
                return _reference_value(m, cat, y, base, heat)

            given = _multiply(
                given,
                multipliers[group],
                ref,
                by,
                delay_to,
                _upper_limit(m, cat),
                (accelerate or {}).get(group),
            )
        if cat["status"] != "lever":
            report.append(row | {"status": "unmapped", "reason": f"{cat['status']}: give a pathway instead"})
            continue
        if not given:
            report.append(row | {"status": "unmapped", "reason": "no values"})
            continue
        if m.unit not in cat["units"]:
            report.append(row | {"status": "unmapped", "reason": f"unit {m.unit!r} not in {cat['units']}"})
            continue
        if isinstance(cat["field"], tuple) and cat["field"][0] == "heat_health":
            if heat is None:
                report.append(row | {"status": "unmapped", "reason": "no heat_health submodel for this city"})
                continue
            attr = cat["field"][1][m.unit]
            neutral = cat.get("neutral", 0.0)
            neutral = float(getattr(heat, neutral) if isinstance(neutral, str) else neutral)
            setattr(heat, attr, _series(given, by, neutral))
            report.append(row | {"status": "mapped", "reason": f"heat_health.{attr}"})
            continue
        if isinstance(cat["field"], tuple) and cat["field"][0] == "microclimate":
            if micro is None or micro.calibration_csv is None:
                report.append(
                    row | {"status": "unmapped", "reason": "no microclimate calibration for this city"}
                )
                continue
            attr = cat["field"][1][m.unit]
            if m.unit == "ha" and not micro.city_area_km2:
                report.append(
                    row | {"status": "unmapped", "reason": "unit 'ha' needs microclimate.city_area_km2"}
                )
                continue
            setattr(micro, attr, _series(given, by, 0.0))
            reason = f"microclimate.{attr}"
            if micro.coupling is not None:
                reason += f" (ignored: city couples the microclimate to {micro.coupling})"
            report.append(row | {"status": "mapped", "reason": reason})
            continue
        if isinstance(cat["field"], tuple):  # energy and supply levers: {group: {year: share}}
            part, attr, group = cat["field"]
            target = demand if part == "energy_demand" else supply
            current = dict(getattr(target, attr) or {})
            current[group] = _series(given, by, 0.0)
            setattr(target, attr, current)
            report.append(row | {"status": "mapped", "reason": f"{part}.{attr}[{group}]"})
            continue
        neutral = {"factor": 1.0, "share_removed": 1.0, "share_time_saved": 1.0, "EUR/trip": 1.0}.get(
            m.unit, 0.0
        )
        if m.measure == "ev_fleet":
            current = getattr(mob, cat["field"]) or {}
            start = current.get(by, 0.0)
        else:
            start = neutral
        series = _series({y: _convert(m.measure, m.unit, v, mob) for y, v in given.items()}, by, start)
        setattr(mob, cat["field"], series)
        status = {"status": "mapped", "reason": cat["field"]}
        if m.exposure is not None:
            if m.exposure.kind not in cat["exposure"]:
                status = {
                    "status": "mapped without exposure",
                    "reason": f"exposure {m.exposure.kind!r} not modelled",
                }
            elif m.exposure.kind == "charged_km_per_car_trip":
                unit_scale = dataclasses.replace(
                    base, mobility=dataclasses.replace(mob, road_pricing_km_scale=1.0)
                )
                mob.road_pricing_km_scale = m.exposure.value / K.charged_km_per_car_trip(unit_scale)
            elif m.exposure.kind == "zone_trip_share":
                mob.area_toll_affected_share = m.exposure.value
        report.append(row | status)
    overrides = dict(getattr(demand, "overrides", {}) or {})
    for p in strategy.pathways:
        given = {y: v for y, v in p.values.items() if v is not None}
        row = {
            "element": p.row,
            "kind": "pathway",
            "role": "stated",
            "effect": "exogenous path",
            "source": p.source.document,
        }
        if not given:
            report.append(row | {"status": "unmapped", "reason": "no values"})
            continue
        group = p.backcast_group or p.row
        if multipliers is not None and group in multipliers:
            ref_pts = (reference_rows or {}).get(p.row) or (
                getattr(base.energy_demand, "overrides", {}) or {}
            ).get(p.row)
            if not ref_pts:
                report.append(
                    row
                    | {
                        "status": "mapped",
                        "reason": "MAED-City row replaced; no reference row, not multiplied",
                    }
                )
                overrides[p.row] = given
                continue
            row["multiplier"] = multipliers[group]
            given = _multiply(
                given,
                multipliers[group],
                lambda y, r=ref_pts: _interp(r, y),
                by,
                delay_to,
                float("inf"),
                (accelerate or {}).get(group),
            )
        overrides[p.row] = given
        report.append(row | {"status": "mapped", "reason": "MAED-City row replaced"})
    for o in strategy.outcomes:
        report.append(
            {
                "element": o.quantity,
                "kind": "outcome",
                "role": "target",
                "effect": "",
                "source": o.source.document,
                "status": "check",
                "reason": "",
            }
        )
    demand.overrides = overrides
    parts = {"mobility": mob, "energy_demand": demand}
    if supply is not None:
        parts["energy_supply"] = supply
    if micro is not None:
        parts["microclimate"] = micro
    if heat is not None:
        parts["heat_health"] = heat
    cfg = dataclasses.replace(base, **parts)
    return cfg, pd.DataFrame(report)


def check_outcomes(results: pd.DataFrame, strategy: StrategyFile) -> pd.DataFrame:
    """Model value against each outcome target; ``results`` is a runner output (index year)."""
    rows = []
    for o in strategy.outcomes:
        if o.quantity not in results.columns:
            rows.append(
                {
                    "quantity": o.quantity,
                    "year": None,
                    "target": None,
                    "model": None,
                    "note": "not a model output",
                }
            )
            continue
        if o.relative_to is not None and o.relative_to not in results.index:
            rows.append(
                {
                    "quantity": o.quantity,
                    "year": None,
                    "target": None,
                    "model": None,
                    "note": f"reference year {o.relative_to} outside the model run",
                }
            )
            continue
        for y, target in o.values.items():
            if target is None or y not in results.index:
                continue
            model = float(results.loc[y, o.quantity])
            if o.relative_to is not None:
                model = model / float(results.loc[o.relative_to, o.quantity]) - 1
            rows.append(
                {
                    "quantity": o.quantity,
                    "year": y,
                    "target": target,
                    "model": model,
                    "gap": model - target,
                    "note": o.source.document,
                }
            )
    return pd.DataFrame(rows)


def strategy_outcomes(model, cfg, pi_scenario: float = 1.0, pkm_method: str = "centroid") -> pd.DataFrame:
    """Outcome quantities of one SD run (index year) for `check_outcomes` and `solve_strategy`: trip shares of
    the residents (internal zone; car = driver and passenger), car trips across the city boundary (origin-
    destination zone), passenger-km shares, emissions (total, transport), final energy, population and the
    per-capita values."""
    mm = model.mobility.MobilityModel(cfg, pi_scenario=pi_scenario)
    mob = mm.run()
    ed = model.energy_demand.run_energy_demand(cfg, mob, pkm_method=pkm_method)
    es = model.energy_supply.run_energy_supply(cfg, ed)
    mods = model.mobility
    trips = mm.trips  # [year, mode, length class, zone, coping type]
    internal = trips[:, :, :, mods.TT_IDX["internal"], :].sum(axis=(2, 3))
    od = trips[:, :, :, mods.TT_IDX["origin destination"], :].sum(axis=(2, 3))
    car = [mods.MODE_IDX["car"], mods.MODE_IDX["car passenger"]]
    years = list(np.asarray(mob["years"]).astype(int))
    pop = np.asarray(mm.population["internal"], dtype=float)
    m = mob[pkm_method] if pkm_method in mob else mob["centroid"]
    out = pd.DataFrame(
        {
            "trip_share_car": internal[:, car].sum(axis=1) / internal.sum(axis=1),
            "trip_share_sustainable": 1 - internal[:, car].sum(axis=1) / internal.sum(axis=1),
            "car_volume_city_boundary": od[:, car].sum(axis=1),
            "car_km_share": np.asarray(m["modal_split"]["car"])
            + np.asarray(m["modal_split"]["car passenger"]),
            "ev_share": np.asarray(mob["ev_share"]),
            "emissions_total": np.asarray(es["total_emissions_kt"]),
            "emissions_transport": np.asarray(es["em_transport"]),
            "final_energy": np.asarray(ed["total_final_energy"]),
            "final_energy_transport": np.asarray(ed["total_transport"]),
            "population": pop,
        },
        index=pd.Index(years, name="year"),
    )
    ef_other = cfg.energy_supply.fuel_ef_other_t_per_gwh / 1000
    out["emissions_buildings"] = (
        np.asarray(ed["hh_heating_fossil"])
        + np.asarray(ed["hh_other_fossil"])
        + np.asarray(ed["service_fossil"])
    ) * ef_other
    # process emissions of industry (kt), where a city add-on reports them
    out["emissions_industry"] = np.asarray(ed["industry_fossil"]) * ef_other + np.asarray(
        ed.get("industry_process_kt", np.zeros(len(years)))
    )
    out["emissions_energy_supply"] = np.asarray(es["em_elec_prod"]) + np.asarray(es["em_dh"])
    # territorial: direct combustion and local generation, without imported electricity
    out["emissions_territorial"] = out["emissions_total"] - np.asarray(es["em_elec_imp"])
    buildings = np.asarray(ed["total_households"]) + np.asarray(ed["total_services"])
    out["fossil_share_building_final_energy"] = (
        np.asarray(ed["hh_heating_fossil"])
        + np.asarray(ed["hh_other_fossil"])
        + np.asarray(ed["service_fossil"])
    ) / buildings
    out["electricity_demand_twh"] = np.asarray(ed["ED_CoreElectricityDemand"]) / 1000
    out["district_heat_demand_twh"] = (
        np.asarray(ed["hh_heating_dh"])
        + np.asarray(ed["hh_other_dh"])
        + np.asarray(ed["service_dh"])
        + np.asarray(ed.get("industry_dh", np.zeros(len(years))))
    ) / 1000
    for q in (
        "emissions_total",
        "emissions_transport",
        "emissions_buildings",
        "emissions_industry",
        "emissions_energy_supply",
        "emissions_territorial",
        "final_energy",
        "final_energy_transport",
    ):
        out[f"{q}_per_capita"] = out[q] / out["population"]
    # heat-health submodel (from its base year; earlier years empty)
    hh_mod = getattr(model, "heat_health", None)
    if hh_mod is not None and getattr(cfg, "heat_health", None) is not None:
        hh = hh_mod.run_heat_health(cfg, population=dict(zip(years, pop, strict=True)))
        cols = ["heat_deaths", "heat_days", "ac_share", "cooled_share", "shaded_share", "hap_level"]
        out = out.join(hh[cols])
        out["heat_deaths_per_100k"] = out["heat_deaths"] / out["population"] * 1e5
    return out


def solve_strategy(
    model, city, strategy: StrategyFile, pi_scenario: float = 1.0, evaluate=None
) -> tuple[dict[str, dict[int, float]], pd.DataFrame]:
    """Common scales of the solved measures that meet the outcome targets of their groups (decision D61).

    One scale per group and year of the group (least squares over the targets of the group in those years,
    within the group's bounds). ``evaluate(cfg) -> DataFrame`` defaults to `strategy_outcomes`. Returns the
    scales and the outcome check of the solved configuration."""
    from scipy.optimize import least_squares

    evaluate = evaluate or (lambda cfg: strategy_outcomes(model, cfg, pi_scenario))
    groups = strategy.solve
    keys = [(g.group, y) for g in groups for y in g.years]
    targets = {o.quantity: o for o in strategy.outcomes}

    def scales_of(x):
        out: dict[str, dict[int, float]] = {}
        for (g, y), v in zip(keys, x, strict=True):
            out.setdefault(g, {})[y] = float(v)
        return out

    def resid(x):
        cfg, _ = apply_strategy(model, city, strategy, scales_of(x))
        res = evaluate(cfg)
        r = []
        for g in groups:
            for q in g.targets:
                o = targets[q]
                for y in g.years:
                    t = o.values.get(y)
                    if t is None:
                        continue
                    v = float(res.loc[y, q])
                    if o.relative_to is not None:
                        v = v / float(res.loc[o.relative_to, q]) - 1
                    r.append(v - t)
        return np.asarray(r)

    lo = [g.bounds[0] for g in groups for _ in g.years]
    hi = [g.bounds[1] for g in groups for _ in g.years]
    x0 = np.clip(np.ones(len(keys)), lo, hi)
    fit = least_squares(resid, x0, bounds=(lo, hi), diff_step=0.05)
    scales = scales_of(fit.x)
    cfg, _ = apply_strategy(model, city, strategy, scales)
    return scales, check_outcomes(evaluate(cfg), strategy)


def outcome_margins(results: pd.DataFrame, strategy: StrategyFile, hard: list[str]) -> pd.DataFrame:
    """Margin of each hard target (outcome quantity in ``hard`` with a ``bound``): >= 0 where the target is met,
    in the target's own unit (change against ``relative_to`` or level)."""
    rows = []
    for o in strategy.outcomes:
        if o.quantity not in hard:
            continue
        if o.bound is None:
            raise ValueError(f"hard target {o.quantity}: the outcome needs a bound ('max' or 'min')")
        for y, t in o.values.items():
            if t is None:
                continue
            v = float(results.loc[y, o.quantity])
            if o.relative_to is not None:
                v = v / float(results.loc[o.relative_to, o.quantity]) - 1
            rows.append(
                {
                    "quantity": o.quantity,
                    "year": y,
                    "target": t,
                    "model": v,
                    "margin": (t - v) if o.bound == "max" else (v - t),
                }
            )
    return pd.DataFrame(rows)


def solve_backcast(
    model,
    city,
    strategy: StrategyFile,
    groups: dict[str, tuple[float, float]],
    hard: list[str],
    objective: str = "deviation",
    scales: dict[str, dict[int, float]] | None = None,
    delay_to: int | None = None,
    reference_rows: dict[str, dict[int, float]] | None = None,
    accelerate: dict[str, int] | None = None,
    pi_scenario: float = 1.0,
    evaluate_many=None,
    x0: dict[str, float] | None = None,
    maxiter: int = 60,
    step: float = 0.02,
) -> dict:
    """Multipliers m of the backcast groups (multiples of the stated level, within ``groups[g] = (lo, hi)``) that
    meet every hard target (decision D64).

    objective "deviation": closest to the strategy, min sum (m - 1)^2; "effort": least effort, min sum m / hi.
    The objective needs no model run; the targets are evaluated with ``evaluate_many(list of multiplier dicts) ->
    list of result tables`` (default: sequential `strategy_outcomes`; pass a parallel map for speed), with forward
    differences of ``step`` for the gradients. Solved measures enter at ``scales`` (their level at m = 1); groups in
    ``accelerate`` read a multiplier above 1 as a faster path (see `apply_strategy`).
    Returns multipliers, margins of the hard targets at the solution, the full check and the solver status."""
    from scipy.optimize import minimize

    names = list(groups)
    lo = np.array([groups[g][0] for g in names])
    hi = np.array([groups[g][1] for g in names])

    def run(mults: dict[str, float]) -> pd.DataFrame:
        cfg, _ = apply_strategy(model, city, strategy, scales, mults, delay_to, reference_rows, accelerate)
        return strategy_outcomes(model, cfg, pi_scenario)

    evaluate_many = evaluate_many or (lambda ms: [run(m) for m in ms])
    cache: dict[tuple, np.ndarray] = {}

    def margins_at(points: list[np.ndarray]) -> list[np.ndarray]:
        todo = [p for p in points if tuple(np.round(p, 10)) not in cache]
        if todo:
            res = evaluate_many([dict(zip(names, map(float, p), strict=True)) for p in todo])
            for p, r in zip(todo, res, strict=True):
                cache[tuple(np.round(p, 10))] = outcome_margins(r, strategy, hard).margin.to_numpy()
        return [cache[tuple(np.round(p, 10))] for p in points]

    def cons(x):
        return margins_at([x])[0]

    def cons_jac(x):
        pts = [x] + [
            x + step * np.eye(len(x))[i] * (1 if x[i] + step <= hi[i] else -1) for i in range(len(x))
        ]
        vals = margins_at(pts)
        g0 = vals[0]
        cols = [(vals[i + 1] - g0) / (step * (1 if x[i] + step <= hi[i] else -1)) for i in range(len(x))]
        return np.column_stack(cols)

    if objective == "deviation":

        def obj(x):
            return float(np.sum((x - 1.0) ** 2))

        def obj_jac(x):
            return 2.0 * (x - 1.0)
    elif objective == "effort":

        def obj(x):
            return float(np.sum(x / hi))

        def obj_jac(x):
            return 1.0 / hi
    else:
        raise ValueError(f"unknown objective {objective!r}")

    start_x = np.clip(np.array([(x0 or {}).get(g, 1.0) for g in names]), lo, hi)
    fit = minimize(
        obj,
        start_x,
        jac=obj_jac,
        method="SLSQP",
        bounds=list(zip(lo, hi, strict=True)),
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    x = np.clip(fit.x, lo, hi)
    mults = dict(zip(names, map(float, x), strict=True))
    res = evaluate_many([mults])[0]
    margins = outcome_margins(res, strategy, hard)
    return {
        "multipliers": mults,
        "margins": margins,
        "check": check_outcomes(res, strategy),
        "results": res,
        "success": bool(fit.success) and bool((margins.margin >= -1e-3).all()),
        "message": str(fit.message),
        "objective": float(obj(x)),
        "evaluations": len(cache),
    }
