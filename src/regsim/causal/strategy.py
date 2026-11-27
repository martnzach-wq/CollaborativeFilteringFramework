"""Local strategies as inputs of the SD model (decision D57).

A strategy (`regsim.config.StrategyFile`) supplies the intensity and exposure of measures and the trajectories of
bundled sectors; the SD model supplies the responses. `apply_strategy` maps the measures onto the levers of a city
configuration and the pathways onto its MAED-City rows, and lists what it cannot map; `check_outcomes` compares the
model with the outcome targets of the strategy. See `docs/causal/transfer_framework.md`.
"""

from __future__ import annotations

import copy
import dataclasses

import pandas as pd

from regsim.config import StrategyFile

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
        }
        for g in ("pv", "wind", "hydro", "district_heating", "electrolysis", "other")
    },
    "road_pricing": {
        "field": "policy_road_pricing",
        "units": ("EUR/km",),
        "exposure": ("charged_km_per_car_trip",),
        "response": "behavioural",
        "status": "lever",
    },
    "area_toll": {
        "field": "policy_area_toll",
        "units": ("EUR/entry",),
        "exposure": ("zone_trip_share",),
        "response": "behavioural",
        "status": "lever",
    },
    "parking_management": {
        "field": "policy_parking_cost_adaptation",
        "units": ("factor", "EUR/trip"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "traffic_calming": {
        "field": "policy_capacity_adaptation",
        "units": ("factor", "share_removed"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "pt_optimisation": {
        "field": "policy_pt_traveltime_adaptation",
        "units": ("factor", "share_time_saved"),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "cycling_optimisation": {
        "field": "cycling_resistance_reduction",
        "units": ("share",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "fifteen_minute_city": {
        "field": "policy_distance_adaptation",
        "units": ("index",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "car_occupancy": {
        "field": "policy_car_passenger_attractiveness",
        "units": ("index",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    "ev_fleet": {
        "field": "ev_pi_trajectory",
        "units": ("percent_of_fleet",),
        "exposure": (),
        "response": "behavioural",
        "status": "lever",
    },
    # microclimate levers (D60): share of the PALM-4U pathway per year, or natural units placed on the land-cover
    # classes in the proportions of the pathway; decoupled from traffic calming unless the city sets a coupling
    "greening": {
        "field": ("microclimate", {"share_of_pathway": "policy_plant_trees", "trees": "plant_trees"}),
        "units": ("share_of_pathway", "trees"),
        "exposure": (),
        "response": "physical",
        "status": "lever",
    },
    "unsealing": {
        "field": ("microclimate", {"share_of_pathway": "policy_unseal", "ha": "unseal_ha"}),
        "units": ("share_of_pathway", "ha"),
        "exposure": (),
        "response": "physical",
        "status": "lever",
    },
    "retrofit": {
        "field": None,
        "units": ("percent_per_year",),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
    },
    "heating_change": {
        "field": None,
        "units": ("units_per_year", "share"),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
    },
    "pv": {"field": None, "units": ("MWp",), "exposure": (), "response": "physical", "status": "bundled"},
    "district_heating": {
        "field": None,
        "units": ("share", "GWh"),
        "exposure": (),
        "response": "technical",
        "status": "bundled",
    },
    "electrolysis": {
        "field": None,
        "units": ("MW",),
        "exposure": (),
        "response": "systemic",
        "status": "bundled",
    },
}


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


def apply_strategy(model, city, strategy: StrategyFile) -> tuple[object, pd.DataFrame]:
    """City configuration with the strategy's measures on the levers and its pathways on the MAED-City rows.

    Returns the configuration and a table of every strategy element with its mapping status; elements without a
    lever, without values, or with an unsupported unit or exposure are listed there and left out."""
    from regsim.causal import sd_knowing as K

    base = model.city(city) if isinstance(city, str) else city
    mob = copy.deepcopy(base.mobility)
    demand = copy.deepcopy(base.energy_demand)
    supply = copy.deepcopy(getattr(base, "energy_supply", None))
    micro = copy.deepcopy(getattr(base, "microclimate", None))
    report = []
    by = strategy.base_year
    for m in strategy.measures:
        cat = CATALOGUE[m.measure]
        row = {"element": m.measure, "kind": "measure", "source": m.source.document}
        given = {y: v for y, v in m.intensity.items() if v is not None}
        if cat["status"] != "lever":
            report.append(row | {"status": "unmapped", "reason": f"{cat['status']}: give a pathway instead"})
            continue
        if not given:
            report.append(row | {"status": "unmapped", "reason": "no values"})
            continue
        if m.unit not in cat["units"]:
            report.append(row | {"status": "unmapped", "reason": f"unit {m.unit!r} not in {cat['units']}"})
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
        row = {"element": p.row, "kind": "pathway", "source": p.source.document}
        if not given:
            report.append(row | {"status": "unmapped", "reason": "no values"})
            continue
        overrides[p.row] = given
        report.append(row | {"status": "mapped", "reason": "MAED-City row replaced"})
    for o in strategy.outcomes:
        report.append(
            {
                "element": o.quantity,
                "kind": "outcome",
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
