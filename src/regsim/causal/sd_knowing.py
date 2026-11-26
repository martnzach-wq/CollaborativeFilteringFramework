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
    "policy_distance_adaptation",
)

# perceived-time series of the active modes that act through a lever's PI and calibration factor
# (decision D41); held together with their lever
LEVER_COMPANIONS = {
    "policy_cycling_attractiveness": ("cycling_resistance_reduction",),
    "policy_capacity_adaptation": ("traffic_calming_time_bonus",),
    "policy_distance_adaptation": ("fifteen_minute_time_bonus",),
}


def _hold_at_start(series):
    """Series held at its 2019 value; per-mode series (dict of dicts) mode by mode."""
    if not series:
        return series
    if isinstance(next(iter(series.values())), dict):
        return {k: _hold_at_start(v) for k, v in series.items()}
    start = series.get(2019, series[min(series)])
    return dict.fromkeys(series, start)


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
        for field in (lever, *LEVER_COMPANIONS.get(lever, ())):
            if hasattr(mob, field):
                setattr(mob, field, _hold_at_start(getattr(mob, field)))
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


# ---- class-R recalibration against the ABM light trajectories (decision D32) --------------

CLASS_R_YEARS = (2030, 2040, 2050)
MODES = ("car", "car passenger", "public transport", "bicycle", "walk")


def class_r_targets(path: str | Path, demonstrator: str, years=CLASS_R_YEARS) -> pd.DataFrame:
    """Passenger-km shares by mode of the ABM light reference and optimised pathways
    (`configs/sd_calibration_targets.csv`) as ``quantity``, ``value`` for `recalibrate_residuals`."""
    t = pd.read_csv(path)
    t = t[(t.demonstrator == demonstrator) & (t.quantity == "pkm") & t.year.isin(years)]
    rows = []
    for (scen, year), g in t.groupby(["scenario", "year"]):
        key = "ref" if scen == "reference" else "opt"
        share = g.set_index("mode").value / g.value.sum()
        rows += [{"quantity": f"{key}_{int(year)}_{m}", "value": float(share.get(m, 0.0))} for m in MODES]
    return pd.DataFrame(rows)


def class_r_run_model(model: Knowing, city: str, years=CLASS_R_YEARS):
    """``run_model(values)`` for `recalibrate_residuals`: attractiveness offsets per mode
    (``offset:<mode>``) and ``mode_change_damping``; returns pkm shares of the BAU (PI = 0)
    and full-policy (PI = 1) runs as ``ref_<year>_<mode>`` and ``opt_<year>_<mode>``."""
    base = model.city(city)

    def run(values: dict) -> pd.Series:
        mob = copy.deepcopy(base.mobility)
        mob.attr_offset = {m: float(values[f"offset:{m}"]) for m in MODES if f"offset:{m}" in values}
        if "mode_change_damping" in values:
            mob.mode_change_damping = max(float(values["mode_change_damping"]), 1e-4)
        c = dataclasses.replace(base, mobility=mob)
        out = {}
        for key, pi in (("ref", 0.0), ("opt", 1.0)):
            split = model.mobility.run_mobility(c, pi_scenario=pi)["centroid"]["modal_split"]
            for y in years:
                for m in MODES:
                    out[f"{key}_{y}_{m}"] = float(split[m][y - 2019])
        return pd.Series(out)

    return run


def recalibrate_class_r(
    model: Knowing,
    city: str,
    targets: pd.DataFrame,
    offset_bounds: tuple[float, float] = (-15.0, 15.0),
    damping_bounds: tuple[float, float] = (0.005, 2.0),
    starts: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit attractiveness offsets (car is the reference mode) and the mode-change damping of a
    city to the ABM light pkm shares: bounded least squares (``scipy.optimize.least_squares``)
    from the configured values and ``starts - 1`` random starts; the best fit is returned in the
    format of `sd_bridge.recalibrate_residuals` (``parameter``, ``value``, ``lower``, ``upper``,
    ``draws``, ``loss``)."""
    from scipy.optimize import least_squares

    params = [f"offset:{m}" for m in MODES[1:]] + ["mode_change_damping"]
    run = class_r_run_model(model, city)
    goal = targets.set_index("quantity").value
    mob = model.city(city).mobility
    lo = np.array([offset_bounds[0]] * 4 + [damping_bounds[0]])
    hi = np.array([offset_bounds[1]] * 4 + [damping_bounds[1]])

    def resid(x):
        out = run(dict(zip(params, x, strict=True)))
        return (out.reindex(goal.index) - goal).to_numpy()

    x_conf = np.clip([mob.attr_offset.get(m, 0.0) for m in MODES[1:]] + [mob.mode_change_damping], lo, hi)
    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        x0 = x_conf if k == 0 else rng.uniform(lo, hi)
        res = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-3)
        if best is None or res.cost < best.cost:
            best = res
    loss = float((best.fun**2).sum())
    return pd.DataFrame(
        {
            "parameter": params,
            "value": best.x,
            "lower": best.x,
            "upper": best.x,
            "draws": starts,
            "loss": loss,
        }
    )


# ---- validation: cross-calibration and uncertainty propagation (protocol steps 5 and validation) ----


def class_r_parameters(model: Knowing, city: str) -> pd.DataFrame:
    """The configured class-R parameters of a city (offsets per mode, damping) as ``parameter``,
    ``class``, ``value`` for `sd_bridge.cross_calibrate`."""
    mob = model.city(city).mobility
    rows = [
        {"parameter": f"offset:{m}", "class": "R", "value": mob.attr_offset.get(m, 0.0)} for m in MODES[1:]
    ]
    rows.append({"parameter": "mode_change_damping", "class": "R", "value": mob.mode_change_damping})
    return pd.DataFrame(rows)


def cross_calibrate_class_r(model: Knowing, cities: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Leave-one-city-out transfer of the class-R parameters (`sd_bridge.cross_calibrate`): for each
    directed pair the root mean square error of the target's ABM light pkm shares with the donor's
    parameters, against the target's own calibration. ``cities``: city -> targets of `class_r_targets`."""
    from regsim.causal.sd_bridge import cross_calibrate

    params = {c: class_r_parameters(model, c) for c in cities}
    observed = {c: t.set_index("quantity").value for c, t in cities.items()}

    def run_model(target: str, values: dict) -> pd.Series:
        own = class_r_parameters(model, target).set_index("parameter").value.to_dict()
        return class_r_run_model(model, target)({**own, **values})

    return cross_calibrate(params, run_model, observed)


def propagate_uncertainty(
    model: Knowing,
    city: str,
    targets: pd.DataFrame,
    n_draws: int = 200,
    seed: int = 0,
    input_sd: dict[str, float] | None = None,
    year: int = 2050,
) -> pd.DataFrame:
    """Protocol step 5: joint draws of the class-R parameters (normal approximation from the
    least-squares fit to ``targets``: covariance s^2 (J'J)^-1) and of class-O inputs (relative
    standard deviations ``input_sd``: OD population, car energy per vehicle-km, load factor, CDD), run
    under PI = 0 and PI = 1. Returns quantiles (5/50/95 %) of the car share of pkm, the final energy
    and the direct and total CO2 emissions in ``year``."""
    input_sd = input_sd or {
        "population_od": 0.25,
        "car_fossil_kwh_per_vkm": 0.10,
        "load_factor": 0.05,
        "cdd": 0.10,
    }
    base = model.city(city)
    params = [f"offset:{m}" for m in MODES[1:]] + ["mode_change_damping"]
    x0 = np.array(
        [base.mobility.attr_offset.get(m, 0.0) for m in MODES[1:]] + [base.mobility.mode_change_damping]
    )
    run = class_r_run_model(model, city)
    goal = targets.set_index("quantity").value

    def resid(x):
        return (run(dict(zip(params, x, strict=True))).reindex(goal.index) - goal).to_numpy()

    r0 = resid(x0)
    jac = np.column_stack(
        [
            (resid(x0 + np.eye(len(x0))[k] * 1e-3 * max(1.0, abs(x0[k]))) - r0)
            / (1e-3 * max(1.0, abs(x0[k])))
            for k in range(len(x0))
        ]
    )
    s2 = float(r0 @ r0) / max(len(r0) - len(x0), 1)
    cov = s2 * np.linalg.pinv(jac.T @ jac)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_draws):
        x = rng.multivariate_normal(x0, cov) if k else x0
        x[-1] = max(x[-1], 1e-3)
        mob = copy.deepcopy(base.mobility)
        mob.attr_offset = {m: float(v) for m, v in zip(MODES[1:], x[:-1], strict=True)}
        mob.mode_change_damping = float(x[-1])
        f = {n: (1.0 if k == 0 else float(np.exp(rng.normal(0.0, sd)))) for n, sd in input_sd.items()}
        mob.population_od = mob.population_od * f["population_od"]
        mob.cooling_degree_days = mob.cooling_degree_days * f["cdd"]
        if mob.cooling_degree_days_trajectory:
            mob.cooling_degree_days_trajectory = {
                y: v * f["cdd"] for y, v in mob.cooling_degree_days_trajectory.items()
            }
        te = base.transport_energy
        if te is not None:
            te = dataclasses.replace(
                te,
                car_fossil_kwh_per_vkm=te.car_fossil_kwh_per_vkm * f["car_fossil_kwh_per_vkm"],
                load_factor=te.load_factor * f["load_factor"],
            )
        c = dataclasses.replace(base, mobility=mob, transport_energy=te)
        for pi in (0.0, 1.0):
            mo = model.mobility.run_mobility(c, pi_scenario=pi)
            ed = model.energy_demand.run_energy_demand(c, mo)
            es = model.energy_supply.run_energy_supply(c, ed)
            i = year - 2019
            split = mo["centroid"]["modal_split"]
            rows.append(
                {
                    "draw": k,
                    "pi": pi,
                    "car_share": float(split["car"][i]),
                    "final_energy": float(ed["total_final_energy"][i]),
                    "co2_direct": float(es["emissions_direct_combustion_kt"][i]),
                    "co2_total": float(es["total_emissions_kt"][i]),
                }
            )
    d = pd.DataFrame(rows)
    q = d.groupby("pi")[["car_share", "final_energy", "co2_direct", "co2_total"]].quantile([0.05, 0.5, 0.95])
    q.attrs.update(city=city, year=year, n_draws=n_draws, class_r_sd=np.sqrt(np.diag(cov)).tolist())
    return q


# ---- per-intervention calibration factors (TR Cal <lever>, decision D38) ---------------------

# ABM light measure -> (SD lever series, TR Cal key)
MEASURE_LEVERS = {
    "traffic calming": ("policy_capacity_adaptation", "capacity"),
    "parking management": ("policy_parking_cost_adaptation", "parking"),
    "area toll": ("policy_area_toll", "area_toll"),
    "road pricing": ("policy_road_pricing", "road_pricing"),
    "cycling optimisation": ("policy_cycling_attractiveness", "cycling"),
    "public transport optimisation": ("policy_pt_traveltime_adaptation", "pt_traveltime"),
    "15-minute city": ("policy_distance_adaptation", "distance"),
    "car occupancy incentives": ("policy_car_passenger_attractiveness", "car_passenger"),
}


def single_measure_effect(
    model: Knowing,
    city,
    lever: str,
    calibration: dict | None = None,
    year: int = 2040,
    metric: str = "car_trips",
) -> float:
    """Relative change of car use in ``year`` when only ``lever`` is applied (PI = 1, all other
    levers held at their 2019 value) against the same run without any lever: the experiment of the
    ABM light single-measure results. ``metric`` "car_trips" counts car (driver) trips, the measure
    of ABM light's "car usage" (D3.3: all measures −37.3 % in 2040, car trips −36 %, car pkm −32 %);
    "car_pkm_share" takes the car share of pkm. ``city`` is a name or a CityConfig."""
    base = model.city(city) if isinstance(city, str) else city
    mob = copy.deepcopy(base.mobility)
    if calibration is not None:
        mob.lever_calibration = calibration
    c = dataclasses.replace(base, mobility=mob)
    others = [lv for lv in LEVERS if lv != lever and getattr(c.mobility, lv, None)]
    single = configure(c, {}, removed=others)
    i = year - 2019
    none = configure(c, {}, removed=[lv for lv in LEVERS if getattr(c.mobility, lv, None)])

    def use(c):
        mm = model.mobility.MobilityModel(c, pi_scenario=1.0)
        split = mm.run()["centroid"]["modal_split"]["car"][i]
        return mm.trips[i, 0].sum() if metric == "car_trips" else split

    return float(use(single) / use(none) - 1)


def recalibrate_levers(
    model: Knowing,
    city: str,
    abm_effects: pd.Series,
    bounds: tuple[float, float] = (0.01, 100.0),
    year: int = 2040,
) -> pd.DataFrame:
    """One calibration factor per intervention: for each ABM light measure mapped in
    `MEASURE_LEVERS`, the factor (log bisection within ``bounds``) at which the SD single-measure
    effect equals the ABM light effect (``abm_effects``: measure -> % change of car share).
    Measures whose target lies outside the reachable range get the nearest bound."""
    base = model.city(city)
    cal = dict(base.mobility.lever_calibration)
    rows = []
    for measure, target_pct in abm_effects.items():
        if measure not in MEASURE_LEVERS:
            continue
        lever, key = MEASURE_LEVERS[measure]
        if not getattr(base.mobility, lever, None):
            continue
        target = target_pct / 100

        def eff(f, lever=lever, key=key):
            return single_measure_effect(model, city, lever, {**cal, key: f}, year)

        lo, hi = np.log(bounds[0]), np.log(bounds[1])
        e_lo, e_hi = eff(np.exp(lo)), eff(np.exp(hi))
        if (e_lo - target) * (e_hi - target) > 0:
            f = np.exp(lo) if abs(e_lo - target) < abs(e_hi - target) else np.exp(hi)
        else:
            for _ in range(30):
                mid = (lo + hi) / 2
                if (eff(np.exp(mid)) - target) * (e_lo - target) > 0:
                    lo = mid
                else:
                    hi = mid
            f = float(np.exp((lo + hi) / 2))
        cal[key] = float(f)
        rows.append(
            {"measure": measure, "key": key, "factor": float(f), "abm": target_pct, "sd": 100 * eff(f)}
        )
    return pd.DataFrame(rows)


def recalibrate_joint(
    model: Knowing,
    city: str,
    pathway_targets: pd.DataFrame,
    abm_effects: pd.Series,
    effect_weight: float = 1.0,
    year: int = 2040,
    factor_bounds: tuple[float, float] = (0.01, 100.0),
    offset_bounds: tuple[float, float] = (-15.0, 15.0),
    damping_bounds: tuple[float, float] = (0.005, 2.0),
    coping_odds_ratio: float | None = None,
    coping_weight: float = 0.1,
    distinction_bounds: tuple[float, float] = (0.0, 1.5),
    logit_scale_bounds: tuple[float, float] = (0.005, 20.0),
    adjustment_bounds: tuple[float, float] = (0.02, 1.0),
) -> pd.DataFrame:
    """Joint fit of the per-intervention factors (log scale), the attractiveness offsets and the
    damping to the ABM light pathways and single-measure effects, since effects fitted one by one
    overshoot in the full package. Bounded least squares from the configured values.

    ``pathway_targets`` has ``quantity`` and ``value``: ``ref_<year>_<mode>`` / ``opt_<year>_<mode>``
    are pkm shares (`class_r_targets`), ``trips_ref_<year>_<mode>`` / ``trips_opt_<year>_<mode>`` trip
    shares (`abm_light_targets`). ``abm_effects`` gives the % change of car trips per measure, indexed
    by measure (at ``year``) or by (measure, year); its residuals are weighted by ``effect_weight``.

    With the logit mode choice (``mode_choice="logit"``, decision D51) the logit scale and the
    adjustment speed are fitted instead of the offsets and the damping.

    With ``coping_odds_ratio`` the coping-type profile strength (``attr_distinction``) is fitted as
    well, with the residual ``coping_weight`` x log(odds ratio of car use of type A in the 2050
    reference / ``coping_odds_ratio``), which keeps the type differences stable (decision D46)."""
    from scipy.optimize import least_squares

    base = model.city(city)
    if not isinstance(abm_effects.index, pd.MultiIndex):
        abm_effects = pd.Series(
            abm_effects.to_numpy(), index=pd.MultiIndex.from_tuples([(ms, year) for ms in abm_effects.index])
        )
    abm_effects = abm_effects[
        [
            ms in MEASURE_LEVERS and bool(getattr(base.mobility, MEASURE_LEVERS[ms][0], None))
            for ms, _ in abm_effects.index
        ]
    ]
    measures = list(dict.fromkeys(ms for ms, _ in abm_effects.index))
    keys = [MEASURE_LEVERS[ms][1] for ms in measures]
    goal = pathway_targets.set_index("quantity").value
    cal0 = base.mobility.lever_calibration
    logit = getattr(base.mobility, "mode_choice", "flow") == "logit"
    if logit:
        # pivot-point logit (D51): log scale and adjustment speed take the place of offsets and damping;
        # constant offsets cancel against the base-year pivot
        core0 = [
            np.log(np.clip(base.mobility.logit_scale, *logit_scale_bounds)),
            np.clip(base.mobility.logit_adjustment, *adjustment_bounds),
        ]
        core_lo = [np.log(logit_scale_bounds[0]), adjustment_bounds[0]]
        core_hi = [np.log(logit_scale_bounds[1]), adjustment_bounds[1]]
    else:
        core0 = [
            *np.clip([base.mobility.attr_offset.get(m, 0.0) for m in MODES[1:]], *offset_bounds),
            np.clip(base.mobility.mode_change_damping, *damping_bounds),
        ]
        core_lo = [offset_bounds[0]] * 4 + [damping_bounds[0]]
        core_hi = [offset_bounds[1]] * 4 + [damping_bounds[1]]
    x0 = np.concatenate([np.log(np.clip([cal0.get(k, 1.0) for k in keys], *factor_bounds)), core0])
    lo = np.concatenate([[np.log(factor_bounds[0])] * len(keys), core_lo])
    hi = np.concatenate([[np.log(factor_bounds[1])] * len(keys), core_hi])
    fit_distinction = coping_odds_ratio is not None
    if fit_distinction:
        x0 = np.append(x0, np.clip(base.mobility.attr_distinction, *distinction_bounds))
        lo, hi = np.append(lo, distinction_bounds[0]), np.append(hi, distinction_bounds[1])
    n_core = len(keys) + len(core0)
    configured = [lv for lv in LEVERS if getattr(base.mobility, lv, None)]

    def unpack(x):
        cal = dict(zip(keys, np.exp(x[: len(keys)]), strict=True))
        core = x[len(keys) : n_core]
        if logit:
            shape = {"logit_scale": float(np.exp(core[0])), "logit_adjustment": float(core[1])}
        else:
            shape = {
                "attr_offset": dict(zip(MODES[1:], map(float, core[:4]), strict=True)),
                "mode_change_damping": float(core[4]),
            }
        distinction = float(x[n_core]) if fit_distinction else base.mobility.attr_distinction
        return cal, shape, distinction

    def city_with(x):
        cal, shape, distinction = unpack(x)
        mob = copy.deepcopy(base.mobility)
        mob.lever_calibration = {**cal0, **cal}
        for k, v in shape.items():
            setattr(mob, k, v)
        mob.attr_distinction = distinction
        return dataclasses.replace(base, mobility=mob)

    def car_trips(c):
        mm = model.mobility.MobilityModel(c, pi_scenario=1.0)
        mm.run()
        return mm.trips[:, 0].sum(axis=(1, 2, 3))

    def resid(x):
        c = city_with(x)
        out = {}
        r_coping = []
        for key, pi in (("ref", 0.0), ("opt", 1.0)):
            mm = model.mobility.MobilityModel(c, pi_scenario=pi)
            split = mm.run()["centroid"]["modal_split"]
            for y in CLASS_R_YEARS:
                trips = mm.trips[y - 2019].sum(axis=(1, 2, 3))
                for k, m in enumerate(MODES):
                    out[f"{key}_{y}_{m}"] = split[m][y - 2019]
                    out[f"trips_{key}_{y}_{m}"] = trips[k] / trips.sum()
            if fit_distinction and key == "ref":
                ratio = model.mobility.car_odds_ratio_type_a(mm.trips[2050 - 2019])
                r_coping = [coping_weight * np.log(ratio / coping_odds_ratio)]
        r_path = pd.Series(out, dtype=float).reindex(goal.index).to_numpy(float) - goal.to_numpy(float)
        none = car_trips(configure(c, {}, removed=configured))
        sim = {
            ms: car_trips(configure(c, {}, removed=[lv for lv in configured if lv != MEASURE_LEVERS[ms][0]]))
            / none
            - 1
            for ms in measures
        }
        r_eff = [effect_weight * (sim[ms][y - 2019] - v / 100) for (ms, y), v in abm_effects.items()]
        return np.concatenate([r_path, r_eff, r_coping])

    res = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=400)
    cal, shape, distinction = unpack(res.x)
    r = res.fun
    n_eff = len(abm_effects)
    coping_resid = r[len(goal) + n_eff :]
    r = r[: len(goal) + n_eff]
    rows = [{"parameter": f"cal:{k}", "value": v} for k, v in cal.items()]
    for k, v in shape.items():
        if isinstance(v, dict):
            rows += [{"parameter": f"offset:{m}", "value": float(w)} for m, w in v.items()]
        else:
            rows.append({"parameter": k, "value": v})
    if fit_distinction:
        rows.append({"parameter": "attr_distinction", "value": distinction})
    out = pd.DataFrame(rows)
    sim_eff = 100 * (r[len(goal) :] / effect_weight) + abm_effects.to_numpy()
    out.attrs.update(
        loss_coping=float((coping_resid**2).sum()),
        loss_pathway=float((r[: len(goal)] ** 2).sum()),
        loss_effects=float((r[len(goal) :] ** 2).sum()),
        effects={f"{ms} {y}": float(v) for (ms, y), v in zip(abm_effects.index, sim_eff, strict=True)},
    )
    return out


# ABM light measure names in the scenario exports -> names of `MEASURE_LEVERS`
ABM_LIGHT_MEASURES = {
    "traffic_calming": "traffic calming",
    "parking_management": "parking management",
    "uvar_pricing": "area toll",
    "road_pricing": "road pricing",
    "cycling_optimization": "cycling optimisation",
    "public_transport_optimization": "public transport optimisation",
    "15min_city": "15-minute city",
    "car_occupancy_bonus": "car occupancy incentives",
}
ABM_LIGHT_MODES = {
    "Car": "car",
    "CarP": "car passenger",
    "PT": "public transport",
    "Bike": "bicycle",
    "Walk": "walk",
}


def abm_light_targets(path: str | Path, years=CLASS_R_YEARS) -> tuple[pd.DataFrame, pd.Series]:
    """Calibration targets from an ABM light scenario export (trips, mean time and distance per
    year, scenario, zone, length class and mode; decision D50): pkm and trip shares by mode of the
    reference and all-measures runs (``ref_``/``opt_`` and ``trips_ref_``/``trips_opt_`` quantities)
    and the % change of car trips of each single measure against the reference, indexed by
    (measure, year)."""
    d = pd.read_csv(path)
    d = d[d["mode"].isin(ABM_LIGHT_MODES)].assign(
        mode=lambda x: x["mode"].map(ABM_LIGHT_MODES), pkm=lambda x: x.trips * x.avg_travel_distance_km
    )
    g = d.groupby(["year", "measure", "mode"])[["trips", "pkm"]].sum()
    rows = []
    for y in years:
        for key, measure in (("ref", "reference"), ("opt", "all_measures")):
            x = g.loc[(str(y), measure)]
            for m in MODES:
                rows.append({"quantity": f"{key}_{y}_{m}", "value": x.pkm[m] / x.pkm.sum()})
                rows.append({"quantity": f"trips_{key}_{y}_{m}", "value": x.trips[m] / x.trips.sum()})
    effects = {}
    for y in years:
        ref = g.loc[(str(y), "reference")].trips["car"]
        for raw, ms in ABM_LIGHT_MEASURES.items():
            if (str(y), raw) in g.index.droplevel("mode"):
                effects[(ms, y)] = 100 * (g.loc[(str(y), raw)].trips["car"] / ref - 1)
    return pd.DataFrame(rows), pd.Series(effects)


# ---- transfer of single-measure effects by semi-elasticities (decision D55) -------------------------------


def charged_km_per_car_trip(city) -> float:
    """Mean km charged by road pricing per car trip in the base year (length per zone and class, OD trips
    capped at ``road_pricing_od_km``, times ``road_pricing_km_scale``)."""
    m = city.mobility
    pops = {"internal": m.population_internal, "origin destination": m.population_od}
    num = den = 0.0
    for z, classes in m.trip_length_by_mode.items():
        for c, lengths in classes.items():
            trips = m.initial_trip_rate[z][c][0] * pops[z]
            km = lengths[0]
            if z == "origin destination" and m.road_pricing_od_km is not None:
                km = min(km, m.road_pricing_od_km)
            num += trips * km * m.road_pricing_km_scale
            den += trips
    return num / den


def measure_intensity(model: Knowing, city, measure: str, year: int = 2040) -> float | None:
    """Intensity of an ABM light measure in its natural unit: EUR per car trip (road pricing, parking),
    EUR per entry (area toll), reduction share (capacity, PT time, cycling resistance), distance adaptation
    (15-minute city) or attractiveness (car occupancy). None where the measure is not configured."""
    m = city.mobility

    def at(series):
        return float(model.mobility._interp_series(series)[year - 2019]) if series else None

    if measure == "road pricing":
        r = at(m.policy_road_pricing)
        return None if r is None else r * charged_km_per_car_trip(city)
    if measure == "area toll":
        return at(m.policy_area_toll) if m.area_toll_applicable else None
    if measure == "parking management":
        a = at(m.policy_parking_cost_adaptation)
        return None if a is None else m.parking_cost_base * (a - 1)
    if measure == "public transport optimisation":
        a = at(m.policy_pt_traveltime_adaptation)
        return None if a is None else 1 - a
    if measure == "traffic calming":
        a = at(m.policy_capacity_adaptation)
        return None if a is None else 1 - a
    if measure == "cycling optimisation":
        return at(m.cycling_resistance_reduction)
    if measure == "15-minute city":
        return at(m.policy_distance_adaptation)
    if measure == "car occupancy incentives":
        return at(m.policy_car_passenger_attractiveness)
    raise KeyError(measure)


def transfer_effects(
    model: Knowing, target, donors: dict, year: int = 2040
) -> tuple[pd.Series, pd.DataFrame]:
    """Single-measure effects (% change of car trips) of ``target`` from the effects of ``donors``
    ({city or CityConfig: Series measure -> %}): the mean over donors of effect / intensity, times the
    target's intensity. Returns the target effects and the donors' semi-elasticities."""

    def cfg(c):
        return model.city(c) if isinstance(c, str) else c

    semi = {}
    for d, eff in donors.items():
        row = {}
        for ms, v in eff.items():
            x = measure_intensity(model, cfg(d), ms, year)
            if x:
                row[ms] = v / x
        semi[d if isinstance(d, str) else d.name] = row
    semi = pd.DataFrame(semi).T
    t = cfg(target)
    out = {}
    for ms in semi.columns:
        x = measure_intensity(model, t, ms, year)
        if x:
            out[ms] = float(semi[ms].mean() * x)
    return pd.Series(out), semi


# ---- choice of transfer donors by mobility profile and settlement structure (decision D56) ----------------

PROFILE_FEATURES = (
    "car_trip_share",
    "pt_trip_share",
    "active_trip_share",
    "car_km_per_trip",
    "od_trip_share",
)


def mobility_profile(model: Knowing, city) -> pd.Series:
    """Base-year mobility profile from the configured trip rates and lengths: trip shares of car, PT and
    active modes, mean car trip length, OD share of trips, and the settlement structure (1 for
    polycentric rural)."""
    c = model.city(city) if isinstance(city, str) else city
    m = c.mobility
    pops = {"internal": m.population_internal, "origin destination": m.population_od}
    trips = np.zeros(5)
    car_km = od = 0.0
    for z, classes in m.initial_trip_rate.items():
        for cl, rates in classes.items():
            t = np.asarray(rates) * pops[z]
            trips += t
            if z == "origin destination":
                od += t.sum()
            lengths = (m.trip_length_by_mode or {}).get(z, {}).get(cl)
            car_km += t[0] * (
                lengths[0]
                if lengths
                else (m.initial_trip_length_internal if z == "internal" else m.initial_trip_length_od)[
                    ("short", "medium", "long").index(cl)
                ]
            )
    total = trips.sum()
    return pd.Series(
        {
            "car_trip_share": trips[0] / total,
            "pt_trip_share": trips[2] / total,
            "active_trip_share": (trips[3] + trips[4]) / total,
            "car_km_per_trip": car_km / trips[0],
            "od_trip_share": od / total,
            "polycentric": float(getattr(m, "spatial_structure", "monocentric") == "polycentric_rural"),
        }
    )


def donor_weights(
    model: Knowing, target, donors: list, bandwidth: float = 1.0, structure_weight: float = 2.0
) -> pd.Series:
    """Kernel weights exp(-(d/h)^2) of ``donors`` for ``target`` on the standardised mobility profile; a
    different settlement structure adds ``structure_weight`` to the distance."""
    names = [d if isinstance(d, str) else d.name for d in donors]
    t_name = target if isinstance(target, str) else target.name
    prof = pd.DataFrame(
        {n: mobility_profile(model, c) for n, c in zip(names + [t_name], donors + [target], strict=True)}
    ).T
    feats = prof[list(PROFILE_FEATURES)]
    z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    d = np.sqrt(((z.loc[names] - z.loc[t_name]) ** 2).sum(axis=1) / len(PROFILE_FEATURES))
    d = d + structure_weight * (prof.loc[names, "polycentric"] != prof.loc[t_name, "polycentric"])
    w = np.exp(-((d / bandwidth) ** 2))
    return w / w.sum()


# 10 % / 90 % quantiles of ABM light effect / transferred effect over the cross test of Tallinn, Naples and
# South Westphalia (23 measure effects, 2040; D56)
TRANSFER_BAND = (0.27, 3.47)


def transfer_effects_weighted(
    model: Knowing, target, donors: dict, year: int = 2040, band=TRANSFER_BAND, **kw
) -> dict:
    """Similarity-weighted single-measure effects of ``target``: ``effects`` (weighted mean of the donors'
    semi-elasticities times the target intensity), ``low`` / ``high`` (the weaker and stronger end of the
    empirical transfer band ``band`` around the effect, widened to the single-donor transfers), ``weights``
    and the single-donor transfers ``single``."""
    names = list(donors)
    w = donor_weights(model, target, names, **kw)
    single = {d: transfer_effects(model, target, {d: donors[d]}, year)[0] for d in names}
    table = pd.DataFrame(single)
    weighted = (table * w).sum(axis=1) / table.notna().mul(w).sum(axis=1)
    return {
        "effects": weighted,
        "low": table.min(axis=1),
        "high": table.max(axis=1),
        "weights": w,
        "single": table,
    }
