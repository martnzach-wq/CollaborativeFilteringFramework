"""Causal edge registry (M7.1): graph operations, evidence updates, mechanism classification, C1 checks."""

import numpy as np
import pandas as pd
import pytest

from regsim.causal.check import agreement_rates, apply_e1, check_edges, write_evidence
from regsim.causal.data import region_means
from regsim.causal.graph import Edge, EdgeRegistry, classify_mechanism
from regsim.registry import IndicatorRegistry


def _edge(i, cause, effect, sign="+", status="prior", **kw):
    return Edge(
        id=i,
        cause=cause,
        effect=effect,
        sign_prior=sign,
        grade="B",
        sources=[{"type": "literature", "ref": "test"}],
        status=status,
        **kw,
    )


@pytest.fixture
def registry():
    return EdgeRegistry(
        schema_version="0.2",
        created="2026-11-23",
        edges=[
            _edge("E1", "a", "b"),
            _edge("E2", "b", "c", sign="-"),
            _edge("E3", "c", "a"),  # cycle
            _edge("E4", "a", "d", status="hypothesis"),
            _edge("E5", "a", "e", mechanism_type="shared_denominator"),
            _edge("E6", "b", "f", sign=None),
        ],
    )


def test_registry_file_validates():
    reg = EdgeRegistry.from_yaml("configs/causal_edges.yaml")
    assert len(reg.edges) == 91 and all(e.sources for e in reg.edges)


def test_descendants_follow_statuses_and_mechanisms(registry):
    assert registry.descendants("a") == ["b", "c", "f"]  # cycle back to a is not repeated
    assert "d" in registry.descendants("a", statuses=("prior", "hypothesis"))
    assert "e" in registry.descendants("a", causal_only=False)
    assert registry.descendants("f") == []


def test_update_evidence_keeps_edges(registry):
    changed = registry.update_evidence(
        [
            {"cause": "a", "effect": "b", "beta_std": -0.11, "t": -4.1, "ci95_std": (-0.16, -0.05)},
            {"cause": "b", "effect": "c", "beta_std": -0.2, "t": -3.0},
            {"cause": "c", "effect": "a", "beta_std": 0.05, "t": 1.0},
            {"cause": "b", "effect": "f", "beta_std": 0.3, "t": 5.0},
        ]
    )
    by_id = {e.id: e for e in registry.edges}
    assert changed == ["E1", "E2"] and len(registry.edges) == 6
    assert by_id["E1"].status == "contested" and by_id["E1"].sources[0].ref == "test"
    assert by_id["E1"].data_evidence.within_region_t == -4.1 and by_id["E1"].data_evidence.tier == "E2"
    assert by_id["E2"].status == "supported"
    assert by_id["E3"].status == "prior" and by_id["E3"].data_evidence.status.startswith("undecided")
    assert by_id["E6"].status == "prior"  # no prior sign: evidence recorded, status unchanged


def test_classify_mechanism():
    ind = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    assert classify_mechanism("gdp_pps_hab", "rd_expenditure_gdp", ind) == "shared_denominator"
    assert classify_mechanism("rd_expenditure_gdp", "gdp_pps_hab", ind) == "shared_denominator"
    assert classify_mechanism("employment_rate", "unemployment_rate", ind) == "accounting_identity"
    assert classify_mechanism("net_migration_rate", "pop_growth_rate", ind) == "accounting_identity"
    assert classify_mechanism("pop_density", "cars_per_1000", ind) == "shared_denominator"  # population stock
    assert classify_mechanism("tertiary_attainment", "lifelong_learning", ind) == "shared_survey_base"
    assert classify_mechanism("poverty_exclusion_risk", "gini", ind) == "shared_survey_base"
    assert classify_mechanism("gdp_pps_hab", "cars_per_1000", ind) == "causal"
    assert classify_mechanism("gdp_pps_hab", "unknown", ind) == "causal"


def _panel(seed=0, n_countries=8, per_country=6, years=range(2010, 2020)):
    """Regions where b rises with a between countries but falls with a within countries."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_countries):
        level = rng.normal(0, 3)
        for r in range(per_country):
            dev = rng.normal()
            for y in years:
                a = level + dev + rng.normal(0, 0.1)
                rows.append(
                    {
                        "geo_id": f"R{c}_{r}",
                        "year": y,
                        "parent_id": f"C{c}",
                        "a": a,
                        "b": 2 * level - dev + rng.normal(0, 0.1),
                        "c": -a + rng.normal(0, 0.1),
                    }
                )
    return pd.DataFrame(rows)


def test_check_edges_between_and_within(registry):
    panel = _panel()
    table = check_edges(registry, panel, min_regions=10).set_index("edge_id")
    assert table.loc["E1", "level"] == "regional"
    assert table.loc["E1", "r_between"] > 0.5 and table.loc["E1", "agree_between"] == True  # noqa: E712
    assert table.loc["E1", "r_within_country"] < -0.5 and table.loc["E1", "agree_within_country"] == False  # noqa: E712
    assert table.loc["E1", "label"] == "consistent"
    assert table.loc["E4", "label"] == "not observed"  # d is not in the panel
    assert table.loc["E6", "label"] in ("no prior", "not observed")
    rates = agreement_rates(table.reset_index())
    assert set(rates.test) == {"between", "within_country"} and (rates.rate.dropna().between(0, 1)).all()


def test_national_fallback_and_min_years(registry):
    regional = _panel()
    regional["c"] = np.nan  # c only national
    national = regional.groupby(["parent_id", "year"], as_index=False)[["a", "b"]].mean()
    national["c"] = -national["a"]
    national["geo_id"] = national["parent_id"]
    table = check_edges(registry, regional, national, min_regions=10, min_countries=5).set_index("edge_id")
    assert table.loc["E2", "level"] == "national" and np.isnan(table.loc["E2", "r_within_country"])
    means = region_means(regional.head(24), ["a"], min_years=5)  # 10, 10 and 4 years of three regions
    assert means["a"].notna().sum() == 2


def test_e1_written_only_without_data_evidence(registry, tmp_path):
    registry.update_evidence([{"cause": "a", "effect": "b", "beta_std": 0.2, "t": 3.0}])
    table = check_edges(registry, _panel(), min_regions=10)
    updated = apply_e1(registry, table)
    by_id = {e.id: e for e in registry.edges}
    assert "E1" not in updated and by_id["E1"].data_evidence.tier == "E2"
    assert by_id["E2"].data_evidence.tier == "E1" and by_id["E2"].data_evidence.between_r is not None
    path = write_evidence(table, tmp_path / "ev.parquet", run="r1")
    write_evidence(table, path, run="r1")  # same run replaces its rows
    write_evidence(table, path, run="r2")
    assert len(pd.read_parquet(path)) == 2 * len(table)


def test_to_yaml_keeps_file_format(tmp_path):
    path = tmp_path / "edges.yaml"
    text = (
        "schema_version: '0.2'\ncreated: '2026-11-23'\nnote_to_reviewers: kept\nedges:\n"
        "- id: E1\n  cause: a\n  effect: b\n  sign_prior: +\n  grade: B\n  data_evidence: null\n"
        "  sources:\n  - type: literature\n    ref: test\n  status: prior\n"
    ).replace("\n", "\r\n")
    path.write_bytes(text.encode())
    reg = EdgeRegistry.from_yaml(path)
    reg.to_yaml(path)
    assert path.read_bytes() == text.encode()  # unchanged registry: identical file
    reg.update_evidence([{"cause": "a", "effect": "b", "beta_std": 0.2, "t": 3.0}])
    reg.to_yaml(path)
    lines = path.read_bytes().decode().split("\r\n")
    assert "note_to_reviewers: kept" in lines and "  status: supported" in lines
    assert lines.index("  data_evidence:") < lines.index("  sources:")  # original key order
