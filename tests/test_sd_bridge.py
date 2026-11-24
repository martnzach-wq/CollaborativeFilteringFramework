"""SD bridge (M7.4) on synthetic inputs: Vensim parsing, perturbation, comparison, levers, transfer protocol."""

import numpy as np
import pandas as pd
import pytest

from regsim.causal import sd_bridge as B

MDL = """{UTF-8}
Car Trips[mode]= INTEG (
	Trip Growth[mode] - Mode Change[mode]\\
		, Initial Trips[mode]) ~~|
Trip Growth[mode]=
	Population * Trip Rate[mode]
	~	trips/year
	~		|
Population=
	1000
	~	persons
	~		|
Trip Rate[car]=
	2 ~~|
Trip Rate[bike]=
	1
	~
	~		|
Initial Trips[mode]=
	Trip Rate[mode] * Population
	~
	~		|
Mode Change[mode]=
	Car Trips[mode] * Fuel Cost
	~
	~		|
Fuel Cost=
	0.1
	~
	~		|
Observed Share[mode]:RAW::=
	Car Trips Data[mode] / Population
	~
	~		|
mode: car, bike
	~
	~		|
\\\\\\---/// Sketch information - do not modify anything except names
10,1,Car Trips,100,100,40,20,3,3,0,0,0,0,0,0
"""


def test_parse_vensim_edges(tmp_path):
    path = tmp_path / "toy.mdl"
    path.write_text(MDL, encoding="utf-8")
    e = B.parse_vensim_edges(path)
    edges = set(zip(e.cause, e.effect, strict=True))
    assert edges == {
        ("trip growth", "car trips"),
        ("mode change", "car trips"),
        ("initial trips", "car trips"),
        ("population", "trip growth"),
        ("trip rate", "trip growth"),
        ("trip rate", "initial trips"),
        ("population", "initial trips"),
        ("car trips", "mode change"),
        ("fuel cost", "mode change"),
    }  # data equation (:=) and subscript range left out; self-references dropped
    t = e.set_index(["cause", "effect"])
    assert t.loc[("trip growth", "car trips"), "effect_is_stock"]
    assert t.loc[("population", "trip growth"), "cause_is_constant"]
    assert t.loc[("fuel cost", "mode change"), "cause_is_constant"]
    assert not t.loc[("trip rate", "trip growth"), "cause_is_constant"]  # element-wise: calibrated data


def _toy(scales, removed=()):
    """Outputs by year from two inputs and two levers."""
    years = np.arange(2019, 2051)
    cost = 1.0 * scales.get("cost", 1.0)
    pop = 100.0 * scales.get("pop", 1.0)
    toll = 0.0 if "toll" in removed else 0.5 * scales.get("toll", 1.0)
    bike = 0.0 if "bike" in removed else 0.1
    car = 1000.0 * cost**-0.3 * (1 - toll) * (1 - bike) * np.ones(len(years))
    return pd.DataFrame({"car_km_cap": car, "active_km_cap": 200 * cost**0.1 * (1 + bike) * np.ones(len(years)),
                         "emissions_total": car * pop / 100}, index=pd.Index(years, name="year"))  # fmt: skip


def test_perturbation_and_levers():
    f = B.perturbation_elasticities(_toy, {}, ["cost", "pop"], rel_step=0.1, year=2050).set_index(
        ["driver", "output"]
    )
    assert f.loc[("cost", "car_km_cap"), "elasticity"] == pytest.approx((1.1**-0.3 - 1) / 0.1)
    c = B.perturbation_elasticities(_toy, {}, ["cost"], ["car_km_cap"], method="central")
    assert c.elasticity.iloc[0] == pytest.approx(-0.3)
    assert f.loc[("pop", "emissions_total"), "elasticity"] == pytest.approx(1.0)
    lv = B.lever_effects(_toy, ["toll", "bike"]).set_index(["lever", "output"])
    assert lv.loc[("toll", "car_km_cap"), "effect"] == pytest.approx(-0.5)
    assert lv.loc[("bike", "active_km_cap"), "effect"] == pytest.approx(0.1)


def test_lever_ranking_agreement():
    t = pd.DataFrame(
        [
            ("A", "road", -0.42), ("A", "cycle", -0.05), ("A", "park", -0.03), ("A", "toll", -0.73),
            ("B", "road", -0.72), ("B", "cycle", -0.21), ("B", "park", -0.01),
        ],
        columns=["city", "lever", "effect"],
    ).assign(output="car_km_cap")  # fmt: skip
    r = B.lever_ranking(t)
    for _city, g in r.groupby("city"):
        assert list(g.sort_values("rank").lever) == ["road", "cycle", "park"]  # toll not in B
    assert r.attrs["spearman"]["A-B"] == pytest.approx(1.0) and r.attrs["max_ratio"] == pytest.approx(4.2)


def test_compare_flags_and_diagnoses():
    e = pd.DataFrame(
        [
            ("T", "fuel", "active", -0.34, "as configured"),
            ("T", "fuel", "active", 0.09, "calibration off"),
            ("T", "population (internal)", "car", -0.58, "as configured"),
            ("T", "population", "car", 0.01, "as configured"),
            ("T", "vot", "car", 0.0, "as configured"),
            ("T", "fuel", "car", -0.28, "as configured"),
            ("T", "cdd", "car", 2.0, "as configured"),
        ],
        columns=["city", "driver", "output", "elasticity", "variant"],
    )
    exp = pd.DataFrame(
        [
            ("fuel", "active", "+", np.nan), ("population (internal)", "car", "0", np.nan),
            ("population", "car", "0", np.nan), ("vot", "car", "+", np.nan), ("fuel", "car", "-", -0.3),
            ("cdd", "car", "+", 0.5),
        ],
        columns=["driver", "output", "sign", "value"],
    )  # fmt: skip
    c = B.compare(e, expectations=exp).set_index(["driver", "output"])
    assert c.loc[("fuel", "active"), "status"] == "model_review"
    assert c.loc[("fuel", "active"), "diagnosis"].startswith("calibration off")
    assert c.loc[("population (internal)", "car"), "diagnosis"].startswith("population scaled consistently")
    assert c.loc[("vot", "car"), "diagnosis"] == "unexplained"
    assert c.loc[("fuel", "car"), "status"] == "consistent"
    assert "factor" in c.loc[("cdd", "car"), "reason"]  # 2.0 against 0.5


def test_map_to_registry_relations():
    m = pd.DataFrame(
        {"sd": ["a", "b", "c"], "regsim": ["x", "y", "z"], "relation": ["same", "proxy", "inverse_proxy"]}
    )
    e = pd.DataFrame(
        {"driver": ["a", "a", "a"], "output": ["b", "c", "missing"], "elasticity": [0.1, 0.2, 0.3]}
    )
    out = B.map_to_registry(e, m)
    assert list(out.relation) == ["proxy", "inverse_proxy"] and list(out.sign_factor) == [1.0, -1.0]


def test_domain_check_and_next_demonstrator():
    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"], index=[f"R{i}" for i in range(40)])
    t = B.domain_check(m, None, ["R0", "R1"], ["a", "b", "c"])
    assert t.loc["R0", "distance_nearest"] == 0 and t.loc["R0", "domain_class"] == "close"
    assert set(t.domain_class) <= {"close", "moderate", "distant", "extrapolation"}
    one = B.domain_check(m, "R5", ["R0", "R1"], ["a", "b", "c"])
    assert one["distance"] == pytest.approx(t.loc["R5", "distance_nearest"])
    n = B.next_demonstrator(m, ["R0", "R1"])
    assert n.median_nearest.iloc[0] <= n.attrs["median_nearest"] and "R0" not in set(n.candidate)
    assert B.report_rules("extrapolation", False) == {"rankings": True, "directions": True, "levels": False}
    assert B.report_rules("moderate", False)["levels"] and B.report_rules("distant", True)["levels"]


def test_recalibration_and_transfer_never_copy_residuals():
    def run_model(x):  # mode shares from two residual constants
        a, b = x["base_car"], x["damping"]
        return pd.Series({"car": a / (a + b), "bike": b / (a + b)})

    targets = pd.DataFrame({"quantity": ["car", "bike"], "value": [0.7, 0.3]})
    cal = B.recalibrate_residuals(
        run_model, {"base_car": 1.0, "damping": 1.0}, targets, ["base_car", "damping"]
    )
    v = dict(zip(cal.parameter, cal.value, strict=True))
    assert (
        v["base_car"] / (v["base_car"] + v["damping"]) == pytest.approx(0.7, abs=1e-3)
        and cal.draws.iloc[0] == 1
    )
    imputed = targets.assign(sd=[0.05, 0.05])
    dist = B.recalibrate_residuals(
        run_model, {"base_car": 1.0, "damping": 1.0}, imputed, ["base_car"], n_draws=20
    )
    assert dist.draws.iloc[0] == 20 and dist.upper.iloc[0] > dist.lower.iloc[0]
    params = pd.DataFrame(
        {"parameter": ["pop", "vot", "base_car", "branch", "toll"], "class": ["O", "B", "R", "S", "P"],
         "value": [1e5, 15.0, 9.9, 1.0, 0.5]}
    )  # fmt: skip
    calls = []

    def recal(names):
        calls.append(names)
        return pd.DataFrame({"parameter": names, "value": [2.0], "lower": [1.5], "upper": [2.5]})

    out = B.transfer_parameters(
        params, "TALLINN", "X", observed={"pop": (5e4, 4e4, 6e4)}, behavioural={"vot": lambda: (6.0, 4.0, 8.0)},
        recalibrate=recal,
    ).set_index("parameter")  # fmt: skip
    assert calls == [["base_car"]] and out.loc["base_car", "value"] == 2.0  # recalibrated, not 9.9
    assert (
        out.loc["pop", "value"] == 5e4
        and out.loc["vot", "upper"] == 8.0
        and out.loc["branch", "value"] == 1.0
    )
    unset = B.transfer_parameters(params, "TALLINN", "X").set_index("parameter")
    assert np.isnan(unset.loc["base_car", "value"])  # without recalibration class R stays unset


def test_cross_calibrate_reports_class_errors():
    cities = {
        "A": pd.DataFrame({"parameter": ["base", "vot"], "class": ["R", "B"], "value": [2.0, 10.0]}),
        "B": pd.DataFrame({"parameter": ["base", "vot"], "class": ["R", "B"], "value": [1.0, 5.0]}),
    }
    own = {c: dict(zip(p.parameter, p.value, strict=True)) for c, p in cities.items()}

    def run_model(city, values):
        x = {**own[city], **values}
        return pd.Series({"car": x["base"] / (x["base"] + 1) + 0.001 * x["vot"]})

    observed = {c: run_model(c, {}) for c in cities}
    out = B.cross_calibrate(cities, run_model, observed)
    assert len(out) == 4 and (out.rmse_own == 0).all()
    r = out.set_index(["donor", "target", "class"])
    assert (
        r.loc[("A", "B", "R"), "rmse"] > r.loc[("A", "B", "B"), "rmse"]
    )  # residual constants do not transfer


def test_parameter_classes_file():
    pc = B.load_parameter_classes()
    by = pc["by_parameter"]
    assert by["mobility.value_of_time_base"] == "B" and by["mobility.mode_change_damping"] == "R"
    assert by["mobility.ev_pi_trajectory"] == "B" and by["mobility.apply_cdd_baseattr_compensation"] == "S"
    assert pc["domain_lens"][0] == "gdp_pps_hab"
