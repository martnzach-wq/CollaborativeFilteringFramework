"""Causal layer M7.3: estimators (D13), partial pooling (D14), direction test, queries (D12), causal lens,
graph-constrained forecast. Synthetic panels only."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from regsim.causal.graph import Edge, EdgeRegistry
from regsim.causal.groups import macro_groups, pooled_effects
from regsim.causal.panel import align, direction_test, fit_edge, reference_offsets, screen
from regsim.causal.queries import edge_effect, vision_check, what_if
from regsim.evaluation.benchmark import graph_forecast_benchmark


def _dynamic(seed=0, G=200, T=15, b0=0.3, b1=0.1, rho=0.5, groups=None):
    """dy = b0 dx + b1 dx_{-1} + rho dy_{-1} + entity drift + noise; ``groups`` scales b0 and b1 per group."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(G):
        scale = 1.0 if groups is None else groups[g % len(groups)]
        a = rng.normal(0, 0.5)
        x = y = dyl = dxl = 0.0
        for t in range(T + 20):
            dx = 0.3 * dxl + rng.normal()
            dy = scale * (b0 * dx + b1 * dxl) + rho * dyl + a + rng.normal()
            x, y, dxl, dyl = x + dx, y + dy, dx, dy
            if t >= 20:
                rows.append((f"g{g}", 2000 + t, f"C{g % 10}", x, y, g % len(groups or [0])))
    return pd.DataFrame(rows, columns=["geo_id", "year", "parent_id", "x", "y", "grp"])


def test_bias_correction_recovers_rho_and_long_run():
    panel = _dynamic(seed=1, G=300)
    plain = fit_edge(panel, "x", "y", bias_correction="none")
    jack = fit_edge(panel, "x", "y")
    gmm = fit_edge(panel, "x", "y", method="gmm")
    assert jack.method == "fe_jackknife" and gmm.method == "gmm" and plain.method == "fe"
    assert plain.rho < 0.45  # Nickell bias of the within estimator with T = 15
    assert abs(jack.rho - 0.5) < abs(plain.rho - 0.5) and abs(gmm.rho - 0.5) < 0.1
    assert abs(jack.long_run_raw - 0.8) < abs(plain.long_run_raw - 0.8)
    assert abs(jack.beta_raw - 0.4) < 0.06


def test_pooled_effects_shrink_towards_mean():
    panel = _dynamic(seed=2, G=240, groups=[0.0, 1.0, 2.0])
    groups = panel.drop_duplicates("geo_id").set_index("geo_id").grp.map({0: "low", 1: "mid", 2: "high"})
    tab, s = pooled_effects(panel, "x", "y", groups, bias_correction="none")
    tab = tab.set_index("group")
    assert tab.loc["high", "beta_raw"] > tab.loc["mid", "beta_raw"] > tab.loc["low", "beta_raw"]
    assert s["I2"] > 0.8 and s["p"] < 0.01 and s["tau2"] > 0
    assert (abs(tab.beta_pooled - s["mu"]) <= abs(tab.beta_raw - s["mu"]) + 1e-12).all()


def test_macro_groups_from_country_codes():
    parents = pd.Series({"S:AT1": "S:AT", "S:EE0": "S:EE", "S:XX1": "S:XX"})
    g = macro_groups(parents, {"West": ["AT"], "North": ["EE"]})
    assert g.to_dict() == {"S:AT1": "West", "S:EE0": "North"}


def test_direction_test_finds_lagged_cause():
    rng = np.random.default_rng(3)
    rows = []
    for g in range(150):
        x = y = 0.0
        dx_prev = 0.0
        for t in range(20):
            dx = rng.normal()
            dy = 0.6 * dx_prev + rng.normal()  # x leads y by one year
            x, y, dx_prev = x + dx, y + dy, dx
            rows.append((f"g{g}", 2000 + t, f"C{g % 8}", x, y))
    panel = pd.DataFrame(rows, columns=["geo_id", "year", "parent_id", "x", "y"])
    r = direction_test(panel, "x", "y", n_tests=10)
    assert r["result"] == "forward" and r["p_forward"] < 1e-6 and r["p_reverse"] > 0.005
    assert direction_test(panel, "y", "x", n_tests=10, events=False)["result"] == "reverse"
    prof = pd.DataFrame(r["event_profile"]).set_index("h")
    # y follows the step in x one year later
    assert prof.loc[1, "beta"] > 3 * prof.loc[1, "se"] and abs(prof.loc[0, "beta"]) < prof.loc[1, "beta"]
    assert r["pre_trend_p"] > 0.01


def test_reference_offsets_and_align(tmp_path):
    path = tmp_path / "ref.yaml"
    path.write_text(
        "indicators:\n  gini: EU-SILC; income reference period mostly t-1\n"
        "  old_age_dependency: 1 January stock\n  gdp: calendar year (flow)\n",
        encoding="utf-8",
    )
    off = reference_offsets(path)
    assert off == {"gini": -1, "old_age_dependency": 0, "gdp": 0}
    panel = pd.DataFrame(
        {"geo_id": "a", "parent_id": "c", "year": [2010, 2011, 2012], "gini": [1.0, 2.0, 3.0]}
    )
    out = align(panel, off).set_index("year")
    assert out.loc[2010, "gini"] == 2.0 and np.isnan(out.loc[2012, "gini"])


def test_screen_marks_artefacts():
    from regsim.registry import IndicatorRegistry

    rng = np.random.default_rng(4)
    rows = []
    for g in range(60):
        e = u = 0.0
        for t in range(15):
            de = rng.normal()
            e, u = e + de, u - de + rng.normal(0, 0.2)  # employment and unemployment move together
            rows.append((f"g{g}", 2000 + t, f"C{g % 6}", e, u))
    panel = pd.DataFrame(
        rows, columns=["geo_id", "year", "parent_id", "employment_rate", "unemployment_rate"]
    )
    ind = IndicatorRegistry.from_yaml("configs/indicators_eurostat_v1.yaml")
    out = screen(panel, ["employment_rate", "unemployment_rate"], registry_indicators=ind)
    assert (
        out.discovered.all()
        and (out.mechanism == "accounting_identity").all()
        and not out.to_literature.any()
    )


def _edge(i, cause, effect, sign="+", status="prior", **kw):
    return Edge(
        id=i,
        cause=cause,
        effect=effect,
        sign_prior=sign,
        grade=kw.pop("grade", "B"),
        sources=[{"type": "literature", "ref": "t"}],
        status=status,
        **kw,
    )


@pytest.fixture
def toy():
    edges = EdgeRegistry(
        schema_version="0.2",
        created="2026-11-23",
        edges=[
            _edge("A", "gdp", "cars"),
            _edge("B", "gdp", "jobs"),
            _edge("C", "jobs", "cars"),
            _edge("D", "cars", "gdp"),  # closes a cycle
            _edge("E", "jobs", "rate", sign="-"),
            _edge("H", "rate", "jobs", sign="-"),  # closes the cycle jobs -> rate -> jobs
            _edge("F", "gdp", "deaths", status="contested"),
            _edge("G", "gdp", "lit", strength_prior={"value": 0.5, "range": [0.3, 0.7]}),
        ],
    )
    effects = pd.DataFrame(
        [
            {"cause": "gdp", "effect": "cars", "group": "all", "beta_pooled": 0.2, "se_pooled": 0.05,
             "long_run_raw": 0.4, "se_long_run_raw": 0.1, "t_cause": "log", "t_effect": "log"},
            {"cause": "gdp", "effect": "cars", "group": "East", "beta_pooled": 0.6, "se_pooled": 0.1,
             "long_run_raw": 0.9, "se_long_run_raw": 0.2, "t_cause": "log", "t_effect": "log"},
            {"cause": "gdp", "effect": "jobs", "group": "all", "beta_pooled": 10.0, "se_pooled": 1.0,
             "long_run_raw": 10.0, "se_long_run_raw": 1.0, "t_cause": "log", "t_effect": "identity"},
            {"cause": "jobs", "effect": "cars", "group": "all", "beta_pooled": 0.01, "se_pooled": 0.0,
             "long_run_raw": 0.01, "se_long_run_raw": 0.0, "t_cause": "identity", "t_effect": "log"},
        ]
    )  # fmt: skip
    values = {"gdp": 20000.0, "cars": 500.0, "jobs": 70.0, "rate": 8.0, "lit": 1.0}
    return edges, effects, values


def test_edge_effect_follows_source_order(toy):
    edges, effects, values = toy
    by_id = {e.id: e for e in edges.edges}
    e = by_id["A"]
    assert edge_effect(e, effects, "East", "short_run", values)["source"] == "panel_group"
    assert edge_effect(e, effects, "West", "short_run", values)["elasticity"] == 0.2  # pooled
    assert edge_effect(e, effects, "West", "long_run", values)["elasticity"] == 0.4
    lit = edge_effect(by_id["G"], effects, None, "long_run", values)
    assert lit["source"] == "literature" and lit["elasticity"] == 0.5 and lit["tier"] == "E1"
    sign = edge_effect(by_id["G"], effects, None, "short_run", values)
    assert sign["source"] == "sign_only" and np.isnan(sign["elasticity"])
    jobs = edge_effect(by_id["B"], effects, None, "short_run", values)
    assert jobs["elasticity"] == pytest.approx(10.0 / 70.0)  # identity effect: beta / value


def test_what_if_propagates_paths_and_leaves_cycles_out(toy):
    edges, effects, values = toy
    out = what_if(None, edges, "x", {"gdp": 0.1}, effects=effects, values=values).set_index("indicator")
    dlog = np.log1p(0.1)
    jobs = 10.0 / 70.0 * dlog
    cars = 0.2 * dlog + 0.01 * 70.0 * jobs  # direct and through jobs
    assert out.loc["jobs", "change"] == pytest.approx(np.expm1(jobs))
    assert out.loc["cars", "change"] == pytest.approx(np.expm1(cars))
    assert out.loc["cars", "edge_ids"] == "A,B,C" and out.loc["cars", "tier"] == "E2"
    assert out.loc["cars", "lower"] < out.loc["cars", "change"] < out.loc["cars", "upper"]
    assert np.isnan(out.loc["rate", "change"]) and out.loc["rate", "sign"] == -1.0  # sign only
    assert "deaths" not in out.index and "F" in out.loc["jobs", "contested"]
    assert out.attrs["cyclic_edges_left_out"] == ["H"]  # D points back to the shocked node, held fixed
    east = what_if(
        None, edges, "x", {"gdp": 0.1}, effects=effects, values=values, groups=pd.Series({"x": "East"})
    )
    assert east.set_index("indicator").loc["cars", "change"] > out.loc["cars", "change"]
    lr = what_if(None, edges, "x", {"gdp": 0.1}, horizon="long_run", effects=effects, values=values)
    assert (lr.horizon == "long_run").all() and lr.set_index("indicator").loc[
        "lit", "sources"
    ] == "literature"


def test_vision_check_flags(toy):
    edges, effects, values = toy
    targets = {"cars": 600.0, "rate": 6.0, "lit": 1.2, "unreached": 1.0}
    out = vision_check(None, edges, "x", targets, 2030, effects=effects, values=values, horizon="short_run")
    cars = out[(out.target == "cars") & (out.lever == "gdp")].iloc[0]
    assert cars.required_change == pytest.approx(np.expm1(np.log(600 / 500) / 0.2)) and not cars.feasible
    jobs = out[(out.target == "cars") & (out.lever == "jobs")].iloc[0]
    assert jobs.required_change == pytest.approx(np.expm1(np.log(600 / 500) / (0.01 * 70)))
    assert out[out.target == "rate"].source.iloc[0] == "sign_only"
    assert (
        out[out.target == "unreached"].mismatch.all() and out[out.target == "cars"].mismatch.eq(False).all()
    )
    lit = vision_check(None, edges, "x", {"lit": 1.2}, 2030, effects=effects, values=values).iloc[0]
    assert lit.source == "literature" and lit.required_change == pytest.approx(np.expm1(np.log(1.2) / 0.5))


def test_s_causal_and_analogues_on_lens():
    from regsim.api.runtime import Fitted
    from regsim.models.hier_mf import HierTypedMF
    from regsim.predict.tasks import analogues
    from regsim.similarity.causal_lens import s_causal
    from regsim.synthetic import generate_hierarchical
    from regsim.uncertainty.conformal import Conformalised

    d = generate_hierarchical(n_per_level=(3, 15, 60), n_indicators=6, rank=3, seed=5).model_data()
    model = Conformalised(partial(HierTypedMF, rank=3, lam_u=0.3)).fit(d)
    c = d.column_ids
    edges = EdgeRegistry(
        schema_version="0.2", created="x", edges=[_edge("A", c[0], c[1]), _edge("B", c[2], c[1])]
    )
    g = [x for x, lv in zip(d.geo_ids, d.levels(), strict=True) if lv == 2][0]
    out = s_causal(Fitted(model, d, None), edges, g, c[1], k=5)
    assert out.attrs["lens"] == [c[0], c[2], c[1]] and len(out) == 5 and 0 <= out.attrs["jaccard"] <= 1
    ana = analogues(model, g, k=5, target=c[1], registry=edges)
    assert list(ana.geo_id) == list(out.geo_id)


def test_graph_constrained_forecast_beats_unconstrained_var():
    rng = np.random.default_rng(6)
    k = 8
    rows = []
    for g in range(80):
        z = np.zeros(k)
        dz_prev = np.zeros(k)
        for t in range(14):
            dz = rng.normal(size=k)
            dz[1] += 0.7 * dz_prev[0]  # the only cross-effect: x0 -> x1
            z = z + dz
            dz_prev = dz
            rows.append((f"g{g}", 2000 + t, *z))
    panel = pd.DataFrame(rows, columns=["geo_id", "year", *[f"x{i}" for i in range(k)]])
    edges = EdgeRegistry(schema_version="0.2", created="x", edges=[_edge("A", "x0", "x1")])
    b = graph_forecast_benchmark(panel, [f"x{i}" for i in range(k)], edges)
    assert len(b) == 2 * k and b.loc[b.indicator == "x1", "n_parents"].iloc[0] == 1
    x1 = b[b.indicator == "x1"]
    assert (x1["graph-constrained VAR"] < x1["own AR(1)"]).all()  # the true cause adds skill
    others = b[b.indicator != "x1"]
    assert np.allclose(others["graph-constrained VAR"], others["own AR(1)"])  # no parents: own lag only
