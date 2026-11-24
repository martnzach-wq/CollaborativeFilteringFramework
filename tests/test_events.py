"""Intervention effects (M7.5): event table, matrix completion, counterfactual effects with placebos."""

import numpy as np
import pandas as pd
import pytest

from regsim.causal.events import effect, load_events, mc_nnm


def _panel(seed=0, size=0.5, start=2018, anticipation=0.0, announce=None, G=100, T=18):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(G, 2)), rng.normal(size=(T, 2))
    Y = rng.normal(size=G)[:, None] + np.cumsum(rng.normal(0, 0.3, T))[None, :] + 0.5 * U @ V.T
    Y = Y + rng.normal(0, 0.2, (G, T))
    sd = Y.std()
    years = np.arange(2005, 2005 + T)
    Y[0, years >= start] += size * sd
    if announce is not None:
        Y[0, (years >= announce) & (years < start)] += anticipation * sd
    panel = pd.DataFrame([(f"R{g}", int(y), Y[g, i]) for g in range(G) for i, y in enumerate(years)],
                         columns=["geo_id", "year", "y"])  # fmt: skip
    return panel, size * sd, anticipation * sd


def test_load_events(tmp_path):
    ev = load_events("configs/events.yaml")
    assert (
        ev.loc[0, "id"] == "EE_vehicle_tax_2025" and ev.loc[0, "announcement_year"] < ev.loc[0, "start_year"]
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "events:\n- {id: x, geo_id: a, intervention: i, announcement_year: 2026, start_year: 2025, source: s}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_events(bad)


def test_mc_nnm_imputes_without_bias():
    panel, _, _ = _panel(seed=3, size=0.0, G=40)
    W = panel.pivot(index="geo_id", columns="year", values="y")
    errors = []
    for g in W.index[:15]:
        M = W.copy()
        M.loc[g, 2018:] = np.nan
        errors.append((W.loc[g, 2018:] - mc_nnm(M).loc[g, 2018:]).to_numpy())
    e = np.concatenate(errors)
    assert abs(e.mean()) < 0.1 and np.sqrt((e**2).mean()) < 0.5


def test_effect_recovers_synthetic_effect():
    panel, true, _ = _panel(seed=1)
    ev = pd.DataFrame([{"id": "e", "geo_id": "R0", "announcement_year": 2018, "start_year": 2018}])
    out = effect(None, ev, "y", analogues=20, placebo=60, panel=panel)
    s = out.attrs["summary"].set_index("phase").loc["treatment"]
    assert s.lower <= true <= s.upper and s.p_value < 0.1
    assert (out.phase == "treatment").all() and out.pre_end.iloc[0] == 2017


def test_announcement_year_shifts_pre_period():
    panel, true, ant = _panel(seed=2, start=2019, announce=2017, anticipation=0.4)
    ev = pd.DataFrame([{"id": "e", "geo_id": "R0", "announcement_year": 2017, "start_year": 2019}])
    out = effect(None, ev, "y", analogues=20, placebo=40, panel=panel)
    assert out.pre_end.iloc[0] == 2016 and set(out[out.phase == "anticipation"].year) == {2017, 2018}
    s = out.attrs["summary"].set_index("phase")
    assert s.loc["anticipation", "att"] > 0 and s.loc["treatment", "att"] > s.loc["anticipation", "att"]
    late = effect(None, ev.assign(announcement_year=2019), "y", analogues=20, placebo=40, panel=panel)
    assert late.pre_end.iloc[0] == 2018  # announcement = start: anticipation years enter the pre-period


def test_event_without_post_data_is_skipped():
    panel, _, _ = _panel(seed=4, size=0.0)
    ev = pd.DataFrame([{"id": "e", "geo_id": "R0", "announcement_year": 2030, "start_year": 2030}])
    out = effect(None, ev, "y", panel=panel, placebo=10)
    assert out.empty and out.attrs["summary"].empty
