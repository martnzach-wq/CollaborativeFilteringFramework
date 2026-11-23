"""Effect similarity (M7.6): varying-coefficient fit, validation gate, region effects, s_effect, views."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import shapely  # noqa: E402

from regsim.similarity.effect import (  # noqa: E402
    effect_distances,
    effect_observations,
    equivalence_probability,
    fit_varying,
    moderators,
    region_effects,
    s_effect,
    validate_gate,
)

ALPHAS = (0.0, 0.3, 3.0, np.inf)


def _panel(slope: float, seed: int = 0, countries: int = 14, per: int = 8, years: int = 14):
    """Cumulative effect 0.3 + slope * z_g, where z_g is the region's mean of the moderator `m`."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(countries):
        for r in range(per):
            z = rng.normal()
            b = 0.3 + slope * z
            x = y = dy = 0.0
            for t in range(years):
                dx = rng.normal()
                dy = b * dx + 0.2 * dy + rng.normal(0, 0.5)
                x, y = x + dx, y + dy
                rows.append((f"S:C{c}R{r}", 2000 + t, f"S:C{c}", x, y, z + rng.normal(0, 0.05)))
    return pd.DataFrame(rows, columns=["geo_id", "year", "parent_id", "x", "y", "m"])


def _mods(panel):
    m = panel.groupby("geo_id")[["m"]].mean()
    return (m - m.mean()) / m.std()


def test_gate_passes_moderated_effect_and_fails_homogeneous():
    het = _panel(0.25)
    parents = het.drop_duplicates("geo_id").set_index("geo_id").parent_id
    g = validate_gate(het, "x", "y", _mods(het), parents, alphas=ALPHAS)
    assert g["heterogeneous"] and g["rel_err"] < 0.6 and g["n_groups"] == 14
    hom = _panel(0.0, seed=1)
    g0 = validate_gate(hom, "x", "y", _mods(hom), parents, alphas=ALPHAS)
    assert not g0["heterogeneous"] and g0["rel_err"] > 0.9


def test_fit_varying_recovers_slope_and_region_effects():
    panel = _panel(0.25, seed=2)
    mods = _mods(panel)
    f = fit_varying(panel, "x", "y", mods, alpha=0.0)
    slope = (f.coef[3] + f.coef[4]) / mods["m"].std()  # per unit of z (moderator standardised)
    assert f.heterogeneous and abs(slope - 0.25 * panel.groupby("geo_id").m.mean().std()) < 0.08
    eff = region_effects(f, mods.index)
    assert set(eff.columns) >= {"geo_id", "link", "beta_std", "lo_std", "hi_std", "heterogeneous", "horizon"}
    assert (eff.lo_std < eff.beta_std).all() and (eff.beta_std < eff.hi_std).all()
    hi, lo = mods["m"].idxmax(), mods["m"].idxmin()
    e = eff.set_index("geo_id").beta_raw
    assert e[hi] > e[lo]
    pooled = fit_varying(panel, "x", "y", mods, alpha=np.inf)
    assert not pooled.heterogeneous and region_effects(pooled, mods.index).beta_raw.nunique() == 1


def test_s_effect_equivalence_and_distance():
    assert equivalence_probability(0.0, 0.01, 0.1) == pytest.approx(1.0)
    assert equivalence_probability(0.5, 0.01, 0.1) == pytest.approx(0.0, abs=1e-9)
    assert 0.3 < equivalence_probability(0.1, 0.05, 0.1) < 0.6
    panel = _panel(0.25, seed=3)
    mods = _mods(panel)
    fits = {
        "x->y": fit_varying(panel, "x", "y", mods, alpha=0.0),
        "x->y pooled": fit_varying(panel, "x", "y", mods, alpha=np.inf),
    }
    hi, lo = mods["m"].idxmax(), mods["m"].idxmin()
    same = s_effect(fits, hi, hi)
    assert same.set_index("link").loc["x->y", "p_equivalent"] == pytest.approx(1.0)
    far = s_effect(fits, hi, lo, delta={"x->y": 0.05}).set_index("link")
    assert far.loc["x->y", "p_equivalent"] < 0.05 and far.loc["x->y", "delta"] == 0.05
    assert far.loc["x->y pooled", "status"] == "no transferable difference"
    assert np.isnan(far.loc["x->y pooled", "p_equivalent"])
    D = effect_distances(fits, [hi, lo])
    assert D.loc[hi, lo] == pytest.approx(abs(far.loc["x->y", "difference"]))


def test_moderators_and_derived_observations():
    panel = _panel(0.1, seed=4, countries=4, per=3, years=5).rename(columns={"x": "gdp_pps_hab"})
    groups = pd.Series({g: ("A" if i % 3 else "B") for i, g in enumerate(panel.geo_id.unique())})
    m = moderators(panel, "y", groups)
    assert {"mean_gdp_pps_hab", "mean_y", "group_B"} <= set(m.columns)  # A is the largest group
    assert abs(m["mean_y"].mean()) < 1e-9
    f = fit_varying(_panel(0.2, seed=5), "x", "y", _mods(_panel(0.2, seed=5)), alpha=0.0)
    eff = region_effects(f, f.moderators.index)
    obs = effect_observations(eff, "2008/2024", "abc")
    assert (obs.indicator_id == "effect:x->y").all() and (obs.scenario == "effect").all()
    assert obs["flags"].str.startswith("heterogeneous").all() and (obs.se > 0).all()


def test_effect_views_render(tmp_path):
    from regsim.viz import effect_views as EV

    panel = _panel(0.25, seed=6, countries=6, per=6)
    mods = _mods(panel)
    fits = {"x->y": fit_varying(panel, "x", "y", mods, alpha=0.0)}
    fits["x->z"] = fit_varying(panel.assign(z=panel.y), "x", "z", mods, alpha=np.inf)
    eff = pd.concat([region_effects(f, mods.index) for f in fits.values()], ignore_index=True)
    ids = list(mods.index)
    geometry = {
        g.split(":")[1]: shapely.box(3e6 + i * 5e4, 2e6, 3e6 + (i + 1) * 5e4, 2.1e6)
        for i, g in enumerate(ids)
    }
    groups = pd.Series({g: g.split("R")[0] for g in ids})
    figs = {
        "pairwise": EV.pairwise_comparison(s_effect(fits, ids[0], ids[1])),
        "space": EV.effect_space(eff, "x->y", "x->z", groups),
        "map": EV.effect_map(eff, "x->y", geometry),
        "overlap": EV.neighbour_overlap(
            pd.DataFrame({"general": [0.1, 0.3], "lens": [0.0, 0.2]}, index=["a", "b"])
        ),
        "strip": EV.effect_profile_strip(eff, groups, mods["m"]),
        "transfer": EV.transfer_matrix(fits, ids[:3], ids[3:6]),
    }
    for name, fig in figs.items():
        path = tmp_path / f"{name}.png"
        fig.savefig(path)
        assert path.stat().st_size > 5_000
    cols = EV.vsup_colours(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.5, 1.0]), 1.0)
    grey = [np.linalg.norm(c - EV.NEUTRAL) for c in cols]
    assert grey[0] > grey[1] > grey[2] > 0  # wider interval -> closer to neutral grey, never fully grey
    assert np.isfinite(EV.vsup_colours(np.array([0.5]), np.array([0.0]), 1.0)).all()  # zero width
