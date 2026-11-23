"""Acceptance tests for milestone M7 (causal layer). Skipped until the sub-milestone starts.

Reference values come from the M7 analysis (causal_test_notes.md, causal_panel_results.csv)
on Eurostat NUTS-2 panels 2008-2024; tolerances allow for data revisions.
"""

import pytest

pytestmark = pytest.mark.skip(reason="M7 not started")


# M7.1 registry and consistency checks
def test_registry_loads_and_validates():
    """configs/causal_edges.yaml validates against EdgeRegistry; every edge has >=1 source."""


def test_contested_edges_kept():
    """update_evidence never deletes edges; GDP -> life expectancy becomes 'contested' with sources intact."""


def test_mechanism_classification():
    """classify_mechanism labels GDP <-> R&D intensity as shared_denominator and
    employment <-> unemployment rate as accounting_identity."""


def test_consistency_check_rates():
    """C1 on region means: between-region sign agreement for signed priors in [0.6, 0.8]
    (M7: 27/38); within-country agreement >= 0.85 (M7: 23/25)."""


# M7.2 literature module
def test_literature_pipeline_offline():
    """With recorded OpenAlex responses and a stub LLM, propose -> search -> extract -> verify -> grade
    reproduces the grade of C30 (household income -> cars, grade A) from fixtures."""


def test_grading_rules():
    """grade() applies DESIGN_WEIGHT and GRADE_RULES; an edge with one cross-sectional study is never above C."""


# M7.3 panel module and queries
def test_fit_edge_reference_value():
    """fit_edge(gdp_pps_hab -> cars_per_1000), NUTS-2, country x year FE: beta_std in [0.06, 0.14], t > 2.5."""


def test_heterogeneity_reference_value():
    """heterogeneity by macro-region for GDP -> cars: I2 >= 0.6, p < 0.05."""


def test_screen_flags_identities():
    """screen() on NUTS-2 flags net migration -> population growth; after classify_mechanism it is
    not passed to the literature module."""


def test_what_if_reports_horizon_and_tier():
    """what_if returns one row per descendant with horizon, tier, edge ids and interval;
    GDP +10 % in EE00 gives cars +0.3 % to +2 % (short run)."""


def test_s_causal_differs_from_general_similarity():
    """s_causal(EE00, cars_per_1000, k=10) overlaps general k=10 neighbours by Jaccard <= 0.3."""


def test_graph_constrained_forecast():
    """Graph-constrained VAR beats unconstrained ridge VAR on median relative RMSE (hold-out last two years)."""


# M7.4 SD bridge
def test_parse_vensim_edges():
    """parse_vensim_edges on Submodel_Mobility_v10.mdl returns 109 edges; on Submodel_EnergySupply_v4.mdl 69."""


def test_sd_compare_flags_review():
    """compare() flags fuel car cost -> active-mode km (-0.34) and population -> total emissions (-0.47) as model_review."""


# M7.5 events
def test_event_effect_placebo():
    """On synthetic data with a known effect of +0.5 SD from year t0, events.effect recovers it within the
    90 % interval and placebo p-value < 0.1; announcement-year handling shifts the pre-period."""


def test_direction_test_reference():
    """direction_test on NUTS-2: GDP <-> cars 'undecided' (same-year responses both ways);
    life expectancy -> old-age dependency 'forward'; series aligned by reference period."""


def test_strict_measure_check():
    """The literature measure check removes the three round-2 hand-check failures and keeps
    the Okun regional panels (GDP -> unemployment stays grade A)."""


def test_registry_schema_v02():
    """Registry schema 0.2 validates: 91 edges; GDP -> life expectancy carries opposite signs by horizon."""


def test_sd_perturbation_protocol():
    """Granollers fuel cost -> car-km elasticity -0.28 +/- 0.05; Tallinn CDD response zero with
    calibration and negative on active km without; consistent population scaling |e| < 0.05."""


# M7.6 effect similarity
def test_effect_gate():
    """Leave-one-country-out gate passes GDP -> life expectancy, GDP -> unemployment, GDP -> cars and
    fails household income -> cars, old-age dependency -> GDP, unemployment -> poverty, employment -> fertility."""


def test_s_effect_equivalence():
    """s_effect(EE00, ES51): P(practical equivalence) for GDP -> life expectancy < 0.3; pooled links reported
    as 'no transferable difference'."""


# M7.4 parameter transfer
def test_parameter_classes_complete():
    """Every CityConfig.mobility field is assigned to exactly one class in configs/sd_parameter_classes.yaml."""


def test_domain_check_reference():
    """domain_check with demonstrators EE00, ES51, ITF3 and the six-indicator lens: share of NUTS-2 regions
    outside the range on >= 1 dimension in [0.75, 0.9]; median nearest distance in [0.8, 0.95]."""


def test_residuals_never_copied():
    """transfer_parameters never copies class-R values from donor to follower; it calls recalibrate_residuals."""


def test_report_rules():
    """report_rules('extrapolation', False) allows rankings but not levels; ('moderate', False) allows levels."""


def test_lever_ranking_agreement():
    """lever_ranking on Tallinn and Granollers: road pricing > cycling attractiveness > parking adaptation in both."""
