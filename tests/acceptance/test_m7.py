"""Acceptance tests for milestone M7 (causal layer). Skipped until the sub-milestone starts.

Reference values come from the M7 analysis (causal_test_notes.md, causal_panel_results.csv)
on Eurostat NUTS-2 panels 2008-2024; tolerances allow for data revisions.
"""

import copy
import os
from pathlib import Path

import pandas as pd
import pytest

from regsim.causal.graph import EdgeRegistry, classify_mechanism
from regsim.registry import IndicatorRegistry

not_started = pytest.mark.skip(reason="sub-milestone not started")
CAUSAL_STORE = Path("data/store_causal")
needs_causal_build = pytest.mark.skipif(
    not (CAUSAL_STORE / "observations").exists(), reason="run `regsim build configs/causal.yaml`"
)
REGISTRY = "configs/causal_edges.yaml"

SD_MODEL = Path(os.environ.get("REGSIM_SD_MODEL", "../martnzach-wq/knowing-integrated-sd-model"))
needs_sd = pytest.mark.skipif(
    not (SD_MODEL / "city_config.py").exists(),
    reason="set REGSIM_SD_MODEL to a checkout of the KNOWING SD model",
)


@pytest.fixture(scope="module")
def sd_runs():
    """Perturbation protocol (2050), comparison and lever effects for Tallinn and Granollers."""
    from regsim.causal import sd_knowing
    from regsim.causal.sd_bridge import compare, load_mapping

    model = sd_knowing.load(SD_MODEL)
    cities = ["TALLINN", "GRANOLLERS"]
    protocol = pd.concat([sd_knowing.perturbation_protocol(model, c) for c in cities], ignore_index=True)
    levers = pd.concat([sd_knowing.lever_table(model, c) for c in cities], ignore_index=True)
    _, expectations = load_mapping()
    return {
        "protocol": protocol,
        "compare": compare(protocol[protocol.year == 2050], expectations=expectations),
        "levers": levers,
    }


@pytest.fixture(scope="module")
def m73():
    """Annual panel, fitted model, region groups and panel effects of `configs/causal.yaml`."""
    import warnings

    from regsim.api.runtime import fit
    from regsim.causal.effects import estimate_effects
    from regsim.causal.run import groups, panel
    from regsim.config import RunConfig

    cfg = RunConfig.from_yaml("configs/causal.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = panel(cfg)
        fitted = fit(cfg)
        g = groups(cfg, p, fitted)
        indicators = IndicatorRegistry.from_yaml(*cfg.data.registries)
        edges = EdgeRegistry.from_yaml(cfg.causal.edges)
        effects = estimate_effects(edges, p, g, indicators, cfg.causal.transforms)
    return {
        "cfg": cfg,
        "panel": p,
        "fitted": fitted,
        "groups": g,
        "indicators": indicators,
        "edges": edges,
        "effects": effects,
    }


@pytest.fixture(scope="module")
def m76(tmp_path_factory):
    """Validation gate and varying-coefficient fits of the seven M7.6 links (`regsim causal effects`)."""
    from regsim.causal.run import run_effects
    from regsim.config import RunConfig

    cfg = RunConfig.from_yaml("configs/causal.yaml")
    cfg.output_dir = str(tmp_path_factory.mktemp("effects"))
    cfg.causal.evidence = str(tmp_path_factory.mktemp("evidence") / "evidence.parquet")
    gate, table, folder = run_effects(cfg, write_store=False)
    import pickle

    with (folder / "fits.pkl").open("rb") as fh:
        fits = pickle.load(fh)
    return {"gate": gate, "table": table, "fits": fits}


# M7.1 registry and consistency checks
def test_registry_loads_and_validates():
    """configs/causal_edges.yaml validates against EdgeRegistry; every edge has >=1 source."""
    reg = EdgeRegistry.from_yaml(REGISTRY)
    assert reg.edges and all(len(e.sources) >= 1 for e in reg.edges)
    assert len({e.id for e in reg.edges}) == len(reg.edges)


def test_contested_edges_kept():
    """update_evidence never deletes edges; GDP -> life expectancy becomes 'contested' with sources intact."""
    reg = EdgeRegistry.from_yaml(REGISTRY)
    edge = reg.edge("gdp_pps_hab", "life_expectancy")[0]
    edge.status, edge.data_evidence = "prior", None
    sources = copy.deepcopy(edge.sources)
    n = len(reg.edges)
    changed = reg.update_evidence(
        [
            {
                "cause": "gdp_pps_hab",
                "effect": "life_expectancy",
                "beta_std": -0.105,
                "t": -4.107,
                "level": "NUTS2",
            }
        ]
    )
    assert edge.id in changed and edge.status == "contested" and edge.sources == sources
    assert len(reg.edges) == n


def test_mechanism_classification():
    """classify_mechanism labels GDP <-> R&D intensity as shared_denominator and
    employment <-> unemployment rate as accounting_identity."""
    ind = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    for a, b in [("gdp_pps_hab", "rd_expenditure_gdp"), ("rd_expenditure_gdp", "gdp_pps_hab")]:
        assert classify_mechanism(a, b, ind) == "shared_denominator"
    for a, b in [("employment_rate", "unemployment_rate"), ("unemployment_rate", "employment_rate")]:
        assert classify_mechanism(a, b, ind) == "accounting_identity"


@needs_causal_build
def test_consistency_check_rates(tmp_path):
    """C1 on region means: between-region sign agreement for signed priors in [0.6, 0.8]
    (M7: 27/38); within-country agreement >= 0.85 (M7: 23/25)."""
    from regsim.causal.check import run_check
    from regsim.config import RunConfig

    cfg = RunConfig.from_yaml("configs/causal.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.causal.evidence = str(tmp_path / "evidence.parquet")
    _, rates, _ = run_check(cfg)
    r = rates[rates.grade == "all"].set_index("test")
    assert 0.6 <= r.loc["between", "rate"] <= 0.8, r
    assert r.loc["within_country", "rate"] >= 0.85, r


# M7.2 literature module
def test_literature_pipeline_offline():
    """With recorded OpenAlex responses and a stub LLM, propose -> search -> extract -> verify -> grade
    reproduces the grade of C30 (household income -> cars, grade A) from fixtures."""
    import json

    from regsim.causal.literature import run_edge
    from regsim.data.http import CachedClient

    fixtures = Path("tests/fixtures/literature")
    answers = json.loads((fixtures / "c30_answers.json").read_text(encoding="utf-8"))

    def stub(prompt: str) -> str:  # recorded answers of the M7 analysis, by study title
        step = (
            "measure_check"
            if "Is the cause measured" in prompt
            else "verify"
            if "First-pass" in prompt
            else "extract"
        )
        return json.dumps(answers[next(t for t in answers if t in prompt)][step])

    edge = pd.Series({"id": "C30", "cause": "hh_income_pps", "effect": "cars_per_1000",
                      "query": "household income car ownership elasticity panel"})  # fmt: skip
    out = run_edge(edge, stub, client=CachedClient(fixtures / "http", offline=True))
    assert out["grade"].iloc[0].grade == "A" and out["grade"].iloc[0].consensus == "+"


def test_grading_rules():
    """grade() applies DESIGN_WEIGHT and GRADE_RULES; an edge with one cross-sectional study is never above C."""
    from regsim.causal.literature import DESIGN_WEIGHT, GRADE_RULES, claim_weight, grade

    assert DESIGN_WEIGHT["quasi_experiment"] == 2.5 and GRADE_RULES["A"]["min_weight"] == 4.0
    for design in ("cross-section regression", "simulation / model", "descriptive / correlation"):
        w = claim_weight(design, "regional", True, "causal claim", "same")
        one = pd.DataFrame([{"edge": "e", "sign": "+", "design": design, "weight": w}])
        assert grade(one).grade.iloc[0] in ("C", "D")


# M7.3 panel module and queries
@needs_causal_build
def test_fit_edge_reference_value(m73):
    """fit_edge(gdp_pps_hab -> cars_per_1000), NUTS-2, country x year FE: beta_std in [0.06, 0.14], t > 2.5."""
    from regsim.causal.panel import fit_edge

    ref = fit_edge(m73["panel"], "gdp_pps_hab", "cars_per_1000", bias_correction="none")  # M7 estimator
    assert 0.06 <= ref.beta_std <= 0.14 and ref.t > 2.5, ref
    jack = fit_edge(m73["panel"], "gdp_pps_hab", "cars_per_1000")  # decision D13 default
    assert jack.beta_std > 0 and jack.rho > ref.rho  # the jackknife removes the downward Nickell bias


@needs_causal_build
def test_heterogeneity_reference_value(m73):
    """heterogeneity by macro-region for GDP -> cars: I2 >= 0.6, p < 0.05."""
    from regsim.causal.groups import macro_groups
    from regsim.causal.panel import heterogeneity

    p = m73["panel"]
    groups = macro_groups(
        p.drop_duplicates("geo_id").set_index("geo_id").parent_id, m73["cfg"].causal.macro_regions
    )
    tab, h = heterogeneity(p, "gdp_pps_hab", "cars_per_1000", groups, bias_correction="none")
    assert h["I2"] >= 0.6 and h["p"] < 0.05, h
    assert tab.set_index("group").beta_std.idxmax() == "East"


@needs_causal_build
def test_screen_flags_identities(m73):
    """screen() on NUTS-2 flags net migration -> population growth; after classify_mechanism it is
    not passed to the literature module."""
    from regsim.causal.panel import screen

    pairs = ["net_migration_rate", "pop_growth_rate", "gdp_pps_hab", "cars_per_1000", "employment_rate"]
    out = screen(m73["panel"], pairs, bias_correction="none", registry_indicators=m73["indicators"])
    row = out[(out.cause == "net_migration_rate") & (out.effect == "pop_growth_rate")].iloc[0]
    assert row.discovered and row.mechanism == "accounting_identity" and not row.to_literature


@needs_causal_build
def test_what_if_reports_horizon_and_tier(m73):
    """what_if returns one row per descendant with horizon, tier, edge ids and interval;
    GDP +10 % in EE00 gives cars +0.3 % to +2 % (short run)."""
    from regsim.api.runtime import resolve_id
    from regsim.causal.queries import what_if

    ee = resolve_id(m73["fitted"].data, "EE00")
    out = what_if(
        m73["fitted"], m73["edges"], ee, {"gdp_pps_hab": 0.10}, effects=m73["effects"], groups=m73["groups"]
    )
    assert {"indicator", "change", "lower", "upper", "horizon", "tier", "edge_ids"} <= set(out.columns)
    assert out.indicator.is_unique and (out.horizon == "short_run").all() and out.tier.notna().all()
    cars = out.set_index("indicator").loc["cars_per_1000"]
    assert 0.003 <= cars.change <= 0.02 and cars.tier == "E2" and "E018" in cars.edge_ids, cars


@needs_causal_build
def test_s_causal_differs_from_general_similarity(m73):
    """s_causal(EE00, cars_per_1000, k=10) overlaps general k=10 neighbours by Jaccard <= 0.3."""
    from regsim.api.runtime import resolve_id
    from regsim.similarity.causal_lens import s_causal

    ee = resolve_id(m73["fitted"].data, "EE00")
    out = s_causal(m73["fitted"], m73["edges"], ee, "cars_per_1000", k=10)
    assert out.attrs["jaccard"] <= 0.3, out.attrs


@needs_causal_build
def test_graph_constrained_forecast(m73):
    """Graph-constrained VAR beats unconstrained ridge VAR on median relative RMSE (hold-out last two years)."""
    from regsim.evaluation.benchmark import graph_forecast_benchmark

    p = m73["panel"]
    indicators = [c for c in p.columns if c not in ("geo_id", "parent_id", "year")]
    b = graph_forecast_benchmark(p, indicators, m73["edges"])
    assert b["graph-constrained VAR"].median() < b["unconstrained VAR (ridge)"].median()


# M7.4 SD bridge
@needs_sd
def test_parse_vensim_edges():
    """parse_vensim_edges on Submodel_Mobility_v10.mdl returns 109 edges; on Submodel_EnergySupply_v4.mdl 69."""
    from regsim.causal.sd_bridge import parse_vensim_edges

    assert len(parse_vensim_edges(SD_MODEL / "Submodel_Mobility_v10.mdl")) == 109
    assert len(parse_vensim_edges(SD_MODEL / "Submodel_EnergySupply_v4.mdl")) == 69


@needs_sd
def test_sd_compare_flags_review(sd_runs):
    """compare() flags fuel car cost -> active-mode km (-0.34) and population -> total emissions (-0.47) as model_review."""
    review = sd_runs["compare"][sd_runs["compare"].status == "model_review"].set_index(
        ["city", "driver", "output"]
    )
    fuel = review.loc[("TALLINN", "fuel car cost per km", "active_km_cap")]
    assert round(fuel.elasticity, 2) == -0.34 and fuel.diagnosis.startswith("calibration off")
    pop = review.loc[("TALLINN", "population (internal)", "emissions_total")]
    assert round(pop.elasticity, 2) == -0.47 and pop.diagnosis.startswith("population scaled consistently")


# M7.5 events
@not_started
def test_event_effect_placebo():
    """On synthetic data with a known effect of +0.5 SD from year t0, events.effect recovers it within the
    90 % interval and placebo p-value < 0.1; announcement-year handling shifts the pre-period."""


@needs_causal_build
def test_direction_test_reference(m73):
    """direction_test on NUTS-2: GDP <-> cars 'undecided' (same-year responses both ways);
    life expectancy -> old-age dependency 'forward'; series aligned by reference period."""
    from regsim.causal.panel import direction_test, reference_offsets

    off = reference_offsets(m73["cfg"].causal.reference_periods)
    n = 48  # Bonferroni over both directions of the 24 prior edges of the M7 analysis
    cars = direction_test(m73["panel"], "gdp_pps_hab", "cars_per_1000", n_tests=n, offsets=off)
    assert cars["result"] == "undecided", cars
    le = direction_test(m73["panel"], "life_expectancy", "old_age_dependency", n_tests=n, offsets=off)
    assert le["result"] == "forward" and le["aligned"], le


def test_strict_measure_check():
    """The literature measure check removes the three round-2 hand-check failures and keeps
    the Okun regional panels (GDP -> unemployment stays grade A)."""
    from regsim.causal.literature import claim_weight, combine_rounds, grade, measure_ok

    r2 = pd.read_csv("docs/causal/lit_r2_claims_graded.csv")
    ok = [
        measure_ok(a, b, c)
        for a, b, c in zip(r2.m_cause_measured, r2.m_effect_measured, r2.m_unit_level_ok, strict=True)
    ]
    r2 = r2.assign(edge=r2.tid, sign=r2.v_sign, design=r2.v_identification, horizon=r2.hz, measure_ok=ok)
    r2["weight"] = [
        claim_weight(
            r.v_identification, r.spatial_level, r.europe_relevant, r.v_claim_strength, r.v_quantity_match
        )
        for r in r2.itertuples()
    ]
    hand = pd.read_csv("docs/causal/literature_r2_handcheck.csv")
    failures = hand[hand.handcheck.str.startswith(("wrong outcome", "distant proxy"))]
    assert len(failures) == 3
    kept = set(r2.loc[r2.measure_ok, "title"])
    assert not kept & set(failures.title)  # all three hand-check failures are removed
    r1 = pd.read_csv("docs/causal/causal_edges_literature.csv")
    r1 = r1.assign(
        edge=r1.cand_id, sign=r1.sign_final, design=r1.identification, horizon="not stated", measure_ok=True
    )
    r1["weight"] = [
        claim_weight(r.identification, r.spatial_level, r.europe_relevant, r.claim_strength, r.quantity_match)
        for r in r1.itertuples()
    ]
    graded = grade(combine_rounds(r1, r2)).set_index("edge")
    assert graded.loc["N_gdp_pp_unempl", "grade"] == "A"  # Okun regional panels kept


def test_registry_schema_v02():
    """Registry schema 0.2 validates: 91 edges; GDP -> life expectancy carries opposite signs by horizon."""
    reg = EdgeRegistry.from_yaml(REGISTRY)
    assert reg.schema_version == "0.2" and len(reg.edges) == 91
    by_horizon = reg.edge("gdp_pps_hab", "life_expectancy")[0].literature_r2["by_horizon"]
    assert by_horizon["short_run"]["sign"] == "-" and by_horizon["long_run"]["sign"] == "+"


@needs_sd
def test_sd_perturbation_protocol(sd_runs):
    """Granollers fuel cost -> car-km elasticity -0.28 +/- 0.05; Tallinn CDD response zero with
    calibration and negative on active km without; consistent population scaling |e| < 0.05."""
    p = sd_runs["protocol"]
    p = p[p.year == 2050].set_index(["city", "variant", "driver", "output"]).elasticity
    assert abs(p[("GRANOLLERS", "as configured", "fuel car cost per km", "car_km_cap")] + 0.28) <= 0.05
    assert abs(p[("TALLINN", "as configured", "cooling degree days", "active_km_cap")]) < 1e-6
    assert p[("TALLINN", "calibration off", "cooling degree days", "active_km_cap")] < -0.05
    for city in ("TALLINN", "GRANOLLERS"):
        for out in ("car_km_cap", "active_km_cap"):
            assert abs(p[(city, "as configured", "population", out)]) < 0.05


# M7.6 effect similarity
@needs_causal_build
def test_effect_gate(m76):
    """Leave-one-country-out gate passes GDP -> life expectancy, GDP -> unemployment, GDP -> cars and
    fails household income -> cars, old-age dependency -> GDP, unemployment -> poverty, employment -> fertility."""
    passes = {"gdp_pps_hab->life_expectancy", "gdp_pps_hab->unemployment_rate", "gdp_pps_hab->cars_per_1000"}
    fails = {
        "hh_income_pps->cars_per_1000",
        "old_age_dependency->gdp_pps_hab",
        "unemployment_rate->poverty_exclusion_risk",
        "employment_rate->fertility_rate",
    }
    gate = m76["gate"].set_index("link")
    assert set(gate.index[gate.heterogeneous]) == passes, gate
    assert set(gate.index[~gate.heterogeneous]) == fails, gate


@needs_causal_build
def test_s_effect_equivalence(m76):
    """s_effect(EE00, ES51): P(practical equivalence) for GDP -> life expectancy < 0.3; pooled links reported
    as 'no transferable difference'."""
    from regsim.similarity.effect import s_effect

    out = s_effect(m76["fits"], "NUTS2024:EE00", "NUTS2024:ES51").set_index("link")
    assert out.loc["gdp_pps_hab->life_expectancy", "p_equivalent"] < 0.3, out
    pooled = out[out.index.isin(m76["gate"].link[~m76["gate"].heterogeneous])]
    assert (pooled.status == "no transferable difference").all() and pooled.p_equivalent.isna().all()


# M7.4 parameter transfer
@needs_sd
def test_parameter_classes_complete():
    """Every CityConfig.mobility field is assigned to exactly one class in configs/sd_parameter_classes.yaml."""
    import dataclasses

    from regsim.causal import sd_knowing
    from regsim.causal.sd_bridge import load_parameter_classes

    model = sd_knowing.load(SD_MODEL)
    fields = {f"mobility.{f.name}" for f in dataclasses.fields(model.city_config.MobilityConfig)}
    by = load_parameter_classes()["by_parameter"]  # raises if a parameter is in two classes
    assert fields <= set(by), sorted(fields - set(by))


@needs_causal_build
def test_domain_check_reference():
    """domain_check with demonstrators EE00, ES51, ITF3 and the six-indicator lens: share of NUTS-2 regions
    outside the range on >= 1 dimension in [0.75, 0.9]; median nearest distance in [0.8, 0.95]."""
    from regsim.causal.run import lens_matrix
    from regsim.causal.sd_bridge import domain_check, load_parameter_classes
    from regsim.config import RunConfig

    lens = load_parameter_classes()["domain_lens"]
    table = domain_check(
        lens_matrix(RunConfig.from_yaml("configs/causal.yaml"), lens), None, ["EE00", "ES51", "ITF3"], lens
    )
    assert 0.75 <= table.outside_any.mean() <= 0.9
    assert 0.8 <= table.distance_nearest.median() <= 0.95


@needs_sd
def test_residuals_never_copied():
    """transfer_parameters never copies class-R values from donor to follower; it calls recalibrate_residuals."""
    import numpy as np

    from regsim.causal import sd_bridge, sd_knowing

    model = sd_knowing.load(SD_MODEL)
    tallinn = model.city("TALLINN").mobility
    by = sd_bridge.load_parameter_classes()["by_parameter"]
    params = pd.DataFrame(
        [
            {"parameter": f, "class": by[f"mobility.{f}"], "value": getattr(tallinn, f)}
            for f in ("population_internal", "value_of_time_base", "mode_change_damping", "attr_distinction",
                      "apply_cdd_baseattr_compensation")
        ]
    )  # fmt: skip
    calls = []

    def recalibrate(names):  # stand-in calibration to a follower target: damping and distinction
        calls.append(names)

        def run(x):
            return pd.Series({"car": x["mode_change_damping"] / (1 + x["attr_distinction"])})

        start = dict.fromkeys(names, 1.0)
        return sd_bridge.recalibrate_residuals(
            run, start, pd.DataFrame({"quantity": ["car"], "value": [0.42]}), names
        )

    out = sd_bridge.transfer_parameters(params, "TALLINN", "GRANOLLERS", recalibrate=recalibrate).set_index(
        "parameter"
    )
    residual = out[out["class"] == "R"]
    assert calls == [list(residual.index)]
    assert not np.isclose(residual.loc["mode_change_damping", "value"], tallinn.mode_change_damping)
    assert out.loc["apply_cdd_baseattr_compensation", "value"] == tallinn.apply_cdd_baseattr_compensation


def test_report_rules():
    """report_rules('extrapolation', False) allows rankings but not levels; ('moderate', False) allows levels."""
    from regsim.causal.sd_bridge import report_rules

    assert report_rules("extrapolation", False) == {"rankings": True, "directions": True, "levels": False}
    assert report_rules("moderate", False)["levels"] and report_rules("extrapolation", True)["levels"]


@needs_sd
def test_lever_ranking_agreement(sd_runs):
    """lever_ranking on Tallinn and Granollers: road pricing > cycling attractiveness > parking adaptation in both."""
    from regsim.causal.sd_bridge import lever_ranking

    levers = sd_runs["levers"]
    ranking = lever_ranking(levers[levers.city.isin(["TALLINN", "GRANOLLERS"])])
    order = ["policy_road_pricing", "policy_cycling_attractiveness", "policy_parking_cost_adaptation"]
    for _city, g in ranking.groupby("city"):
        ranks = g.set_index("lever")["rank"]
        assert ranks[order[0]] < ranks[order[1]] < ranks[order[2]], g
