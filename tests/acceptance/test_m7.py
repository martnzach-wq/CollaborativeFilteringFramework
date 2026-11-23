"""Acceptance tests for milestone M7 (causal layer). Skipped until the sub-milestone starts.

Reference values come from the M7 analysis (causal_test_notes.md, causal_panel_results.csv)
on Eurostat NUTS-2 panels 2008-2024; tolerances allow for data revisions.
"""

import copy
from pathlib import Path

import pytest

from regsim.causal.graph import EdgeRegistry, classify_mechanism
from regsim.registry import IndicatorRegistry

not_started = pytest.mark.skip(reason="sub-milestone not started")
CAUSAL_STORE = Path("data/store_causal")
needs_causal_build = pytest.mark.skipif(
    not (CAUSAL_STORE / "observations").exists(), reason="run `regsim build configs/causal.yaml`"
)
REGISTRY = "configs/causal_edges.yaml"


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
@not_started
def test_literature_pipeline_offline():
    """With recorded OpenAlex responses and a stub LLM, propose -> search -> extract -> verify -> grade
    reproduces the grade of C30 (household income -> cars, grade A) from fixtures."""


@not_started
def test_grading_rules():
    """grade() applies DESIGN_WEIGHT and GRADE_RULES; an edge with one cross-sectional study is never above C."""


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
@not_started
def test_parse_vensim_edges():
    """parse_vensim_edges on Submodel_Mobility_v10.mdl returns 109 edges; on Submodel_EnergySupply_v4.mdl 69."""


@not_started
def test_sd_compare_flags_review():
    """compare() flags fuel car cost -> active-mode km (-0.34) and population -> total emissions (-0.47) as model_review."""


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


@not_started
def test_strict_measure_check():
    """The literature measure check removes the three round-2 hand-check failures and keeps
    the Okun regional panels (GDP -> unemployment stays grade A)."""


def test_registry_schema_v02():
    """Registry schema 0.2 validates: 91 edges; GDP -> life expectancy carries opposite signs by horizon."""
    reg = EdgeRegistry.from_yaml(REGISTRY)
    assert reg.schema_version == "0.2" and len(reg.edges) == 91
    by_horizon = reg.edge("gdp_pps_hab", "life_expectancy")[0].literature_r2["by_horizon"]
    assert by_horizon["short_run"]["sign"] == "-" and by_horizon["long_run"]["sign"] == "+"


@not_started
def test_sd_perturbation_protocol():
    """Granollers fuel cost -> car-km elasticity -0.28 +/- 0.05; Tallinn CDD response zero with
    calibration and negative on active km without; consistent population scaling |e| < 0.05."""


# M7.6 effect similarity
@not_started
def test_effect_gate():
    """Leave-one-country-out gate passes GDP -> life expectancy, GDP -> unemployment, GDP -> cars and
    fails household income -> cars, old-age dependency -> GDP, unemployment -> poverty, employment -> fertility."""


@not_started
def test_s_effect_equivalence():
    """s_effect(EE00, ES51): P(practical equivalence) for GDP -> life expectancy < 0.3; pooled links reported
    as 'no transferable difference'."""


# M7.4 parameter transfer
@not_started
def test_parameter_classes_complete():
    """Every CityConfig.mobility field is assigned to exactly one class in configs/sd_parameter_classes.yaml."""


@not_started
def test_domain_check_reference():
    """domain_check with demonstrators EE00, ES51, ITF3 and the six-indicator lens: share of NUTS-2 regions
    outside the range on >= 1 dimension in [0.75, 0.9]; median nearest distance in [0.8, 0.95]."""


@not_started
def test_residuals_never_copied():
    """transfer_parameters never copies class-R values from donor to follower; it calls recalibrate_residuals."""


@not_started
def test_report_rules():
    """report_rules('extrapolation', False) allows rankings but not levels; ('moderate', False) allows levels."""


@not_started
def test_lever_ranking_agreement():
    """lever_ranking on Tallinn and Granollers: road pricing > cycling attractiveness > parking adaptation in both."""
