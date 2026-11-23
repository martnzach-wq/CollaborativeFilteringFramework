"""Literature module (M7.2): weights, grading, strict measure check, offline pipeline with a stub LLM."""

import json
from pathlib import Path

import pandas as pd
import pytest

from regsim.causal import literature as L
from regsim.data.http import CachedClient

FIXTURES = Path("tests/fixtures/literature")


class StubLLM:
    """Answers from ``c30_answers.json`` by study title; records the prompts."""

    model = "stub"

    def __init__(self):
        self.answers = json.loads((FIXTURES / "c30_answers.json").read_text(encoding="utf-8"))
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        step = (
            "measure_check"
            if "Is the cause measured" in prompt
            else "verify"
            if "First-pass extraction" in prompt
            else "extract"
        )
        title = next(t for t in self.answers if t in prompt)
        return "```json\n" + json.dumps(self.answers[title][step]) + "\n```"


def _round1() -> pd.DataFrame:
    c = pd.read_csv("docs/causal/causal_edges_literature.csv")
    c = c.assign(
        edge=c.cand_id, sign=c.sign_final, design=c.identification, horizon="not stated", measure_ok=True
    )
    c["weight"] = [
        L.claim_weight(
            r.identification, r.spatial_level, r.europe_relevant, r.claim_strength, r.quantity_match
        )
        for r in c.itertuples()
    ]
    return c


def _round2() -> pd.DataFrame:
    c = pd.read_csv("docs/causal/lit_r2_claims_graded.csv")
    c = c.assign(edge=c.tid, sign=c.v_sign, design=c.v_identification, horizon=c.hz)
    c["weight"] = [
        L.claim_weight(
            r.v_identification, r.spatial_level, r.europe_relevant, r.v_claim_strength, r.v_quantity_match
        )
        for r in c.itertuples()
    ]
    return c


def test_claim_weight_tables_and_labels():
    assert L.claim_weight("panel_fe", "regional", True, "causal", "exact") == 2.0
    assert L.claim_weight(
        "experiment/RCT", "national", False, "causal claim", "close proxy"
    ) == pytest.approx(3.0 * 0.6 * 0.8)
    assert L.claim_weight(
        "cross_section", "city", True, "authors disclaim causality", "distant proxy"
    ) == pytest.approx(1.0 * 0.8 * 0.3 * 0.4)
    with pytest.raises(ValueError):
        L.claim_weight("anecdote", "regional", True, "causal", "exact")
    recorded = pd.read_csv("docs/causal/causal_edges_literature.csv").weight
    assert (_round1().weight - recorded).abs().max() < 1e-9  # the recorded weights of round 1 reproduce
    r2 = _round2()
    assert (r2.weight - r2.w).abs().max() < 1e-9  # the recorded weights of round 2 reproduce


def test_grade_reproduces_round_one():
    ours = L.grade(_round1()).set_index("edge")
    ref = pd.read_csv("docs/causal/causal_edges_prior.csv").set_index("cand_id")
    ref = ref[ref.grade != "identity"]
    got = ours.grade.reindex(ref.index).fillna("D")
    assert (got == ref.grade).all(), ref.grade[got != ref.grade]
    assert ours.loc["C30", "grade"] == "A" and ours.loc["C30", "n_panel"] == 6


def test_grade_reproduces_round_two_with_horizons():
    r2 = _round2()
    targets = pd.read_csv("docs/causal/causal_edges_literature_r2.csv").set_index("target_id")
    ours = L.grade(L.combine_rounds(_round1(), r2)).set_index("edge").reindex(targets.index)
    for col in ("grade", "sr_grade", "lr_grade"):
        assert (ours[col].fillna("D") == targets[col].fillna("D")).all(), col
    c12 = ours.loc["C12"]
    assert (c12.sr_consensus, c12.lr_consensus) == ("-", "+")  # GDP -> life expectancy by horizon


def test_grading_rules_cap_single_cross_section():
    one = pd.DataFrame(
        [
            {
                "edge": "e",
                "sign": "+",
                "design": "cross-section regression",
                "weight": 1.0,
                "horizon": "long run",
            }
        ]
    )
    assert L.grade(one).grade.iloc[0] == "C"
    many = pd.concat([one] * 6, ignore_index=True).assign(
        weight=1.0
    )  # weight 6, no panel or quasi-experiment
    assert L.grade(many).grade.iloc[0] == "B"
    panels = many.assign(design="panel with fixed effects / GMM")
    assert L.grade(panels).grade.iloc[0] == "A"
    conflict = pd.concat([panels, panels.assign(sign="-")], ignore_index=True)
    assert L.grade(conflict).grade.iloc[0] == "D"
    nulls = pd.concat(
        [panels, panels.assign(sign="none")], ignore_index=True
    )  # null results lower consistency
    assert L.grade(nulls).consistency.iloc[0] == pytest.approx(0.5)


def test_measure_check_rule():
    r2 = _round2()
    rule = [
        L.measure_ok(a, b, c)
        for a, b, c in zip(r2.m_cause_measured, r2.m_effect_measured, r2.m_unit_level_ok, strict=True)
    ]
    assert rule == r2.measure_ok.tolist() and sum(rule) == 84
    assert not L.measure_ok("same quantity", "different quantity", True)
    assert not L.measure_ok("same quantity", "acceptable proxy", False)


def test_parse_propose_and_queries():
    assert L.parse_json('```json\n{"a": 1}\n```') == {"a": 1} and L.parse_json(" [1, 2] ") == [1, 2]
    ind = pd.DataFrame({"indicator_id": ["a", "b", "c"], "label": ["A", "B", "C"]})
    known = pd.DataFrame({"cause": ["a"], "effect": ["b"]})
    answer = json.dumps(
        [
            {"cause": "a", "effect": "b", "sign": "+"},  # known
            {"cause": "b", "effect": "c", "sign": "-", "query": "b c"},
            {"cause": "b", "effect": "zz", "sign": "+"},  # unknown indicator
        ]
    )
    log = []
    out = L.propose_edges(ind, known, lambda p: answer, log)
    assert (
        list(zip(out.cause, out.effect, strict=True)) == [("b", "c")] and (out.status == "hypothesis").all()
    )
    assert log[0]["step"] == "propose" and "Known" in log[0]["prompt"]
    flagged = pd.DataFrame(
        [
            {
                "cause": "gdp_pps_hab",
                "effect": "life_expectancy",
                "status": "contested",
                "cause_label": "GDP",
            },
            {"cause": "gdp_pps_hab", "effect": "unemployment_rate", "status": "data_flagged"},
        ]
    )
    qs = L.targeted_queries(flagged)
    assert any("procyclical mortality" in q for q in qs) and any("Okun" in q for q in qs)
    assert any("short run" in q for q in qs if q.startswith("GDP")) and any("long run" in q for q in qs)


def test_pipeline_offline_reproduces_c30():
    edge = pd.Series(
        {"id": "C30", "cause": "hh_income_pps", "effect": "cars_per_1000",
         "query": "household income car ownership elasticity panel"}
    )  # fmt: skip
    client = CachedClient(FIXTURES / "http", offline=True)
    llm = StubLLM()
    out = L.run_edge(edge, llm, client=client)
    assert len(out["hits"]) == 6 and len(out["claims"]) == 6
    assert out["claims"].agree_sign.all() and out["claims"].measure_ok.all()
    g = out["grade"].iloc[0]
    assert g.grade == "A" and g.consensus == "+" and g.n_panel == 6
    assert (
        set(out["log"].step) == {"extract", "verify", "measure_check"} and (out["log"].model == "stub").all()
    )
    assert len(llm.prompts) == 18


def test_llm_provider_is_configurable():
    from regsim.causal.run import load_callable

    assert load_callable("json:dumps") is json.dumps
    with pytest.raises(SystemExit):
        load_callable("json")  # not module:callable
