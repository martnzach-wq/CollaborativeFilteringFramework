"""Views of effect similarity (M7.6, docs/causal/effect_similarity.md, visualisation catalogue).

Every view shows uncertainty: intervals, ellipses or a value-suppressing palette. Inputs are the
outputs of `regsim.similarity.effect` (``s_effect`` tables, ``region_effects`` tables, fits).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse

from regsim.api.runtime import short_id
from regsim.viz.views import CATEGORICAL, EXTENT, GREY, _rings

Z90 = 1.6448536269514722
NEUTRAL = np.array(to_rgb("#d9d9d9"))


def _label(link: str, labels: dict | None) -> str:
    cause, effect = link.split("->")
    lab = labels or {}
    return f"{lab.get(cause, cause)} → {lab.get(effect, effect)}"


def pairwise_comparison(table: pd.DataFrame, names: dict | None = None, labels: dict | None = None):
    """Pairwise effect comparison: each link's standardised effect with 90 % interval for both regions,
    the probability of practical equivalence, pooled links greyed (table from `s_effect`)."""
    a, b = table.attrs.get("geo_a", "A"), table.attrs.get("geo_b", "B")
    names = names or {}
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(table) + 1.5))
    y = np.arange(len(table))[::-1]
    for yi, r in zip(y, table.itertuples(), strict=True):
        het = r.status == "heterogeneous"
        for off, val, se, col in (
            (0.12, r.effect_a, r.se_a, CATEGORICAL[0]),
            (-0.12, r.effect_b, r.se_b, CATEGORICAL[1]),
        ):
            half = Z90 * se if het and np.isfinite(se) else 0.0
            ax.errorbar(val, yi + off, xerr=half, fmt="o", color=col if het else GREY, capsize=3)
        txt = f"P(equivalent) = {r.p_equivalent:.2f}" if het else "no transferable difference"
        ax.text(1.02, yi, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y, [_label(lnk, labels) for lnk in table.link])
    ax.set_xlabel("standardised short-run effect (90 % interval)")
    ax.plot([], [], "o", color=CATEGORICAL[0], label=names.get(a, short_id(a)))
    ax.plot([], [], "o", color=CATEGORICAL[1], label=names.get(b, short_id(b)))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Effect comparison (δ = {table.delta.iloc[0]:g} SD)", fontsize=10)
    fig.tight_layout()
    return fig


def effect_space(
    effects: pd.DataFrame,
    link_x: str,
    link_y: str,
    groups: pd.Series | None = None,
    labels: dict | None = None,
):
    """Effect space: predicted effects of two links per region with 90 % ellipses (the two links are
    fitted separately, so the ellipses are axis-aligned), coloured by region group."""
    e = effects.pivot_table(index="geo_id", columns="link", values=["beta_std", "lo_std", "hi_std"])
    fig, ax = plt.subplots(figsize=(7, 6))
    g = groups.reindex(e.index) if groups is not None else pd.Series("all", index=e.index)
    colours = {lab: CATEGORICAL[i % len(CATEGORICAL)] for i, lab in enumerate(sorted(g.dropna().unique()))}
    for gid in e.index:
        x, y = e.loc[gid, ("beta_std", link_x)], e.loc[gid, ("beta_std", link_y)]
        w = e.loc[gid, ("hi_std", link_x)] - e.loc[gid, ("lo_std", link_x)]
        h = e.loc[gid, ("hi_std", link_y)] - e.loc[gid, ("lo_std", link_y)]
        col = colours.get(g.get(gid), GREY)
        ax.add_patch(Ellipse((x, y), w, h, facecolor=col, alpha=0.08, edgecolor="none"))
        ax.plot(x, y, "o", ms=2.5, color=col)
    for lab, col in colours.items():
        ax.plot([], [], "o", color=col, label=lab)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(_label(link_x, labels))
    ax.set_ylabel(_label(link_y, labels))
    ax.legend(fontsize=8, title="region group", title_fontsize=8)
    ax.set_title("Effect space (standardised effects, 90 % ellipses)", fontsize=10)
    return fig


def vsup_colours(values: np.ndarray, widths: np.ndarray, vmax: float, n_levels: int = 4) -> np.ndarray:
    """Value-suppressing palette: a diverging colour for the value, blended towards a neutral grey in
    ``n_levels`` steps of uncertainty (interval width relative to the widest interval)."""
    top = np.nanmax(widths) if np.size(widths) else 0.0
    u = np.clip(widths / top, 0, 1) if top > 0 else np.zeros(np.shape(widths))
    u = np.minimum(np.floor(u * n_levels), n_levels - 1) / n_levels  # quantised uncertainty levels
    return _vsup(values, u, vmax)


def _vsup(values: np.ndarray, u: np.ndarray, vmax: float) -> np.ndarray:
    """Diverging colour of ``values`` blended towards neutral grey by the uncertainty share ``u``."""
    cmap = plt.get_cmap("RdBu_r")
    v = np.clip(np.asarray(values) / vmax, -1, 1) * 0.5 + 0.5
    base = np.array([cmap(x)[:3] for x in v])
    u = np.asarray(u, dtype=float)
    return base * (1 - u[:, None]) + NEUTRAL * u[:, None]


def effect_map(effects: pd.DataFrame, link: str, geometry: dict, labels: dict | None = None):
    """Effect map with a value-suppressing palette: diverging colours centred at zero, paler where the
    90 % interval is wide. ``geometry`` maps codes without scheme prefix to shapes."""
    e = effects[effects.link == link].assign(code=lambda t: t.geo_id.map(short_id)).set_index("code")
    vmax = float(np.nanmax(np.abs(e.beta_std))) or 1.0
    cols = vsup_colours(e.beta_std.to_numpy(), (e.hi_std - e.lo_std).to_numpy(), vmax)
    colour = dict(zip(e.index, cols, strict=True))
    verts, faces = [], []
    for code, geom in geometry.items():
        for ring in _rings(geom):
            verts.append(ring)
            faces.append(colour.get(code, to_rgb(GREY)))
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [5, 1]})
    ax.add_collection(PolyCollection(verts, facecolors=faces, edgecolors="white", linewidths=0.1))
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    het = bool(e.heterogeneous.iloc[0]) if len(e) else False
    ax.set_title(_label(link, labels) + ("" if het else " (pooled: no transferable difference)"), fontsize=10)
    vals = np.linspace(-vmax, vmax, 9)
    for k, u in enumerate(np.linspace(0, 1, 4, endpoint=False)):  # legend: value x uncertainty level
        lax.scatter(vals, np.full(9, k), c=_vsup(vals, np.full(9, u), vmax), s=120, marker="s")
    lax.set_yticks(range(4), ["certain", "", "", "uncertain"])
    lax.set_xticks([-vmax, 0, vmax], [f"{-vmax:.2f}", "0", f"{vmax:.2f}"])
    lax.set_title("effect × uncertainty", fontsize=8)
    for s in lax.spines.values():
        s.set_visible(False)
    return fig


def neighbour_overlap(overlaps: pd.DataFrame):
    """Neighbour comparison: Jaccard overlap of the 10 nearest regions in effect space with the general
    and the causal-lens neighbours (rows: regions; columns: comparisons)."""
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(overlaps) + 1.5))
    overlaps.plot.barh(ax=ax, color=CATEGORICAL[: overlaps.shape[1]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jaccard overlap of the 10 nearest regions")
    ax.set_title("Effect neighbours versus other neighbours", fontsize=10)
    fig.tight_layout()
    return fig


def effect_profile_strip(
    effects: pd.DataFrame, groups: pd.Series | None = None, order_by: pd.Series | None = None, labels=None
):
    """Effect profile strip: regions × heterogeneous links, ordered by region group and ``order_by``
    (e.g. GDP); colours as in the effect map (value-suppressing)."""
    e = effects[effects.heterogeneous]
    links = sorted(e.link.unique())
    val = e.pivot_table(index="geo_id", columns="link", values="beta_std")[links]
    wid = (e.assign(w=e.hi_std - e.lo_std)).pivot_table(index="geo_id", columns="link", values="w")[links]
    key = pd.DataFrame(index=val.index)
    key["g"] = groups.reindex(val.index).fillna("") if groups is not None else ""
    key["o"] = order_by.reindex(val.index) if order_by is not None else 0.0
    idx = key.sort_values(["g", "o"]).index
    vmax = float(np.nanmax(np.abs(val.to_numpy()))) or 1.0
    img = np.stack(
        [vsup_colours(val.loc[idx, k].to_numpy(), wid.loc[idx, k].to_numpy(), vmax) for k in links], axis=0
    )
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(links) + 1.5))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(links)), [_label(k, labels) for k in links], fontsize=8)
    bounds = key.loc[idx, "g"].to_numpy()
    starts = [0] + [i for i in range(1, len(bounds)) if bounds[i] != bounds[i - 1]]
    ax.set_xticks(starts, [bounds[i] for i in starts], fontsize=7, rotation=90)
    for s in starts[1:]:
        ax.axvline(s - 0.5, color="black", lw=0.6)
    ax.set_title("Effect profiles (regions ordered by group and GDP; pale = uncertain)", fontsize=10)
    fig.tight_layout()
    return fig


def transfer_matrix(fits: dict, pioneers: list[str], followers: list[str], delta: float = 0.1, names=None):
    """Transfer matrix: probability of practical equivalence, averaged over the heterogeneous links,
    between pioneer regions (rows) and follower regions (columns)."""
    from regsim.similarity.effect import s_effect

    names = names or {}
    M = np.full((len(pioneers), len(followers)), np.nan)
    for i, a in enumerate(pioneers):
        for j, b in enumerate(followers):
            t = s_effect(fits, a, b, delta)
            p = t.p_equivalent.dropna()
            M[i, j] = p.mean() if len(p) else np.nan
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(followers), 1 + 0.5 * len(pioneers)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Greens")
    ax.set_xticks(
        range(len(followers)), [names.get(g, short_id(g)) for g in followers], rotation=60, fontsize=8
    )
    ax.set_yticks(range(len(pioneers)), [names.get(g, short_id(g)) for g in pioneers], fontsize=8)
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean P(practical equivalence)")
    ax.set_xlabel("follower")
    ax.set_ylabel("pioneer")
    ax.set_title("Transfer matrix", fontsize=10)
    fig.tight_layout()
    return fig


def _rgb(c) -> str:
    r, g, b = (int(round(255 * x)) for x in c[:3])
    return f"rgb({r},{g},{b})"


def pairwise_plotly(table: pd.DataFrame, names: dict | None = None, labels: dict | None = None):
    """Interactive pairwise effect comparison (see `pairwise_comparison`)."""
    import plotly.graph_objects as go

    names = names or {}
    a, b = table.attrs.get("geo_a", "A"), table.attrs.get("geo_b", "B")
    fig = go.Figure()
    ylab = [_label(lnk, labels) for lnk in table.link]
    for geo, col, side, off in ((a, CATEGORICAL[0], "a", 0.15), (b, CATEGORICAL[1], "b", -0.15)):
        key = f"effect_{side}"
        het = table.status == "heterogeneous"
        half = np.where(het & np.isfinite(table[f"se_{side}"]), Z90 * table[f"se_{side}"], 0.0)
        fig.add_trace(
            go.Scatter(
                x=table[key],
                y=np.arange(len(table))[::-1] + off,
                mode="markers",
                marker={"size": 9, "color": [_rgb(col) if h else GREY for h in het]},
                error_x={"type": "data", "array": half, "visible": True},
                name=names.get(geo, short_id(geo)),
                hovertext=ylab,
            )
        )
    for i, r in enumerate(table.itertuples()):
        txt = (
            f"P(equivalent) {r.p_equivalent:.2f}"
            if r.status == "heterogeneous"
            else "no transferable difference"
        )
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="left", y=len(table) - 1 - i, text=txt, showarrow=False
        )
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(
        yaxis={"tickvals": list(range(len(table)))[::-1], "ticktext": ylab},
        xaxis_title="standardised short-run effect (90 % interval)",
        margin={"r": 190, "l": 10, "t": 40},
        title=f"Effect comparison (δ = {table.delta.iloc[0]:g} SD)" if len(table) else "Effect comparison",
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def effect_map_plotly(
    effects: pd.DataFrame, link: str, geometry: dict, names: dict | None = None, labels=None
):
    """Interactive effect map with the value-suppressing palette (see `effect_map`)."""
    import plotly.graph_objects as go

    names = names or {}
    e = effects[effects.link == link].assign(code=lambda t: t.geo_id.map(short_id)).set_index("code")
    vmax = float(np.nanmax(np.abs(e.beta_std))) or 1.0
    cols = vsup_colours(e.beta_std.to_numpy(), (e.hi_std - e.lo_std).to_numpy(), vmax)
    fig = go.Figure()
    for (code, r), c in zip(e.iterrows(), cols, strict=True):
        if code not in geometry:
            continue
        xs, ys = [], []
        for ring in _rings(geometry[code]):
            xs += [*ring[:, 0], None]
            ys += [*ring[:, 1], None]
        name = names.get(r.geo_id, code)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                mode="lines",
                line={"width": 0.3, "color": "white"},
                fillcolor=_rgb(c),
                hoverinfo="text",
                text=f"{name}<br>effect {r.beta_std:.3f} (90 %: {r.lo_std:.3f} to {r.hi_std:.3f})",
                showlegend=False,
            )
        )
    het = bool(e.heterogeneous.iloc[0]) if len(e) else False
    fig.update_layout(
        title=_label(link, labels) + ("" if het else " (pooled: no transferable difference)"),
        xaxis={"visible": False, "range": EXTENT[:2]},
        yaxis={"visible": False, "range": EXTENT[2:], "scaleanchor": "x"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        plot_bgcolor="white",
    )
    return fig


def event_plot(
    panel: pd.DataFrame, outcome: str, effects: pd.DataFrame, names: dict | None = None, label=None
):
    """Intervention effect (M7.5): observed series of the treated entity, counterfactual from the
    announcement year with the placebo interval, anticipation years shaded (``effects`` from
    `regsim.causal.events.effect`, one event)."""
    names = names or {}
    geo = effects.geo_id.iloc[0]
    obs = panel[panel.geo_id == geo].set_index("year")[outcome].dropna()
    e = effects.sort_values("year")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        obs.index, obs.values, "o-", color=CATEGORICAL[0], label=f"observed: {names.get(geo, short_id(geo))}"
    )
    cf_lo, cf_hi = e.observed - e.upper, e.observed - e.lower  # counterfactual band from the effect interval
    ax.fill_between(
        e.year, cf_lo, cf_hi, color=CATEGORICAL[2], alpha=0.25, label="counterfactual, 90 % placebo interval"
    )
    ax.plot(e.year, e.counterfactual, "s--", color=CATEGORICAL[2], label="counterfactual")
    ant = e[e.phase == "anticipation"].year
    if len(ant):
        ax.axvspan(
            ant.min() - 0.5, ant.max() + 0.5, color=GREY, alpha=0.5, lw=0, label="anticipation (announced)"
        )
    ax.axvline(e.pre_end.iloc[0] + 0.5, color="black", lw=0.6)
    ax.set_xlabel("year")
    ax.set_ylabel(label or outcome)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{e.event.iloc[0]}: effect by counterfactual matrix completion", fontsize=10)
    fig.tight_layout()
    return fig
