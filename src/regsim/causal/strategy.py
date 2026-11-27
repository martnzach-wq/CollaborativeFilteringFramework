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


def _scale_at(scale: dict[int, float], year: int, base_year: int) -> float:
    pts = sorted({base_year: 0.0, **scale}.items())
    return float(np.interp(year, [a for a, _ in pts], [b for _, b in pts]))


def apply_strategy(
    model, city, strategy: StrategyFile, scales: dict[str, dict[int, float]] | None = None
) -> tuple[object, pd.DataFrame]:
    """City configuration with the strategy's measures on the levers and its pathways on the MAED-City rows.

    Stated measures enter with their intensity. Solved measures (decision D61) enter with their shape scaled
    by the common scale of their group, ``scales[group][year]`` (linear between the years, 0 in the base year),
    as found by `solve_strategy`; without a scale they are reported as "to solve" and left out.
    Returns the configuration and a table of every strategy element with its role and mapping status; elements
    without a lever, without values, or with an unsupported unit or exposure are listed there and left out."""
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
        row = {"element": m.measure, "kind": "measure", "role": m.role, "source": m.source.document}
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
        row = {"element": p.row, "kind": "pathway", "role": "stated", "source": p.source.document}
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
                "role": "target",
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
    out["emissions_industry"] = np.asarray(ed["industry_fossil"]) * ef_other
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
