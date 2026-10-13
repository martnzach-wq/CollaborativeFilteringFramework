import pandas as pd
import pytest

from regsim.config import KoppenAdjacency, Rule, RuleSet
from regsim.rules.engine import check_expression, evaluate, gate_summary, koppen_adjacency, load_rules


@pytest.fixture
def rules():
    return load_rules("configs/rules_d36.yaml")


@pytest.fixture
def targets():
    return pd.DataFrame(
        {
            "coastal_type": ["0", "1", None, "2"],
            "mountain_type": [None, "1", "1", "4"],
            "degurba": ["3", "1", "2", None],
            "elevation_min": [120.0, 2.0, 60.0, None],
            "climate_zone_classification_1990_2020_majority": ["Cfa", "BSk", "Dfb", None],
            "annual_precipitation_1990_2020_mean": [0.5, 2.0, 9.0, None],
        },
        index=["a", "b", "c", "e"],
    )


def test_all_d36_gates_expressible(rules):
    assert [r.id[:4] for r in rules.rules] == [f"G{i:03d}" for i in range(1, 11)]
    assert sum(r.severity_published == "hard" for r in rules.rules) == 1


def test_expression_language_rejects_code():
    for bad in ["__import__('os')", "d.x.y == 1", "open('f')", "[x for x in d]", "lambda: 1"]:
        with pytest.raises(ValueError):
            check_expression(bad)
    with pytest.raises(ValueError):
        Rule(id="x", when="exec('1')")


def test_gates_and_missing_values(rules, targets):
    demo = pd.Series(
        {
            "coastal_type": "1",
            "degurba": "1",
            "climate_zone_classification_1990_2020_majority": "Cfa",
            "annual_precipitation_1990_2020_mean": 2.0,
        }
    )
    coastal = evaluate(rules, demo, targets, cic="coastal")
    assert coastal["G001_coastal_eligibility"].tolist() == [True, False, False, False]
    assert coastal["G002_mountain_coastal_incompatibility"].tolist() == [False, True, True, False]
    assert coastal["G009_coastal_cic_high_elevation_target"].tolist() == [True, False, True, False]
    assert coastal["G005_koppen_distant"].tolist() == [False, True, True, False]
    assert "G007_urban_cic_rural_target" not in coastal
    agro = evaluate(rules, demo, targets, cic="agroforestry")
    assert agro["G006_arid_humid_incompatibility"].tolist() == [False, False, True, False]


def test_severity_by_mode(rules, targets):
    demo = pd.Series({"coastal_type": 1})
    flags = evaluate(rules, demo, targets, cic="coastal")
    assert gate_summary(flags, rules, mode="regsim").eligible.all()  # decision D9
    d36 = gate_summary(flags, rules, mode="d36")
    assert d36.eligible.tolist() == [False, True, True, True]
    table10 = gate_summary(flags, rules, mode="d36", severity="table10")
    assert table10.eligible.tolist() == [False, False, False, True]  # G002 and G009 are hard in Table 10
    assert d36.gates_fired["a"].startswith("G001")


def test_koppen_adjacency_and_coverage():
    table = KoppenAdjacency(observed_zones=["Cfa"], adjacent=[("Cfa", "Dfa")])
    assert koppen_adjacency(table, "Dfa", "Cfa") == "adjacent"
    assert koppen_adjacency(table, "Cfa", "Cfb") == "distant"
    assert koppen_adjacency(table, "Dfb", "Dwd") == "adjacent"
    assert koppen_adjacency(table, "Dfb", None) is None
    rs = RuleSet(rules=[Rule(id="cov", when="min_tier_coverage(d, t) < 0.7")])
    t = pd.DataFrame(index=["x", "y", "z"])
    flags = evaluate(rs, pd.Series(dtype=object), t, coverage=pd.Series({"x": 0.5, "y": 0.9}))
    assert flags["cov"].tolist() == [True, False, False]
