"""Interactive explorer (decision D7: Dash) with a linked map and embedding.

Selecting regions (lasso or box) on the map highlights them in the embedding and vice versa; a lasso or box
replaces the selection, a click adds the region. The selection is listed by name and can be cleared. A click also
alternates between setting the reference and the comparison region (both also found by searching their names); the
reference shows its nearest regions, and the difference profile (V6) compares it with the comparison region (by
default its nearest region). Lens, level, number of types and colouring are controls. Distances and types are computed in the full space.

Embedding and trajectories share one coordinate system when indicator directions are shown: a PCA of the
standardised completed profiles of both period windows (model of all windows, ``panel``) for the current lens
and level, into which the regions of the current model and the trajectories are projected. The arrows (indicator
loadings) are therefore the same in both plots. Without directions, the embedding is a PCA of the whitened
profiles (the similarity space) or UMAP, and states its trustworthiness.

Run with ``regsim app CONFIG`` or ``create_dash(fitted, geometry, ...).run()``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import pandas as pd

from regsim.api.runtime import Fitted, short_id
from regsim.config import CAPITAL_REGIONS

CAPTION = {"fontSize": "12px", "color": "#555", "margin": "2px 0 8px 0"}
SOURCES = {"eurostat": "Eurostat", "d36": "KNOWING D3.6", "ardeco": "JRC ARDECO"}
PARTNER_COLOUR = "#c92a2a"
SOURCE_COLOURS = {"eurostat": "#1f5fa8", "d36": "#2e8b3a", "ardeco": "#8e44ad"}
SHORTEN = {"Transport Infrastructure – ": "Transport: ", "Climate Zone Classification": "Climate zone"}
HELP = {
    "intro": "Explore how similar regions are, which types they form and how they moved between the two period "
    "windows. Values come from the fitted model; unobserved values are predictions with intervals.",
    "lens": "Indicators that define similarity, types and the embedding (empty = all). [source] shows where an "
    "indicator comes from; the buttons select all indicators of one source.",
    "sources": "Source of an indicator: the primary source of its values; copies from other sources (e.g. "
    "JRC ARDECO for Eurostat indicators) fill gaps.",
    "types": "Number of types of the typology (auto = chosen by bootstrap stability and BIC).",
    "colour": "Colour of map and embedding: type, similarity to the reference (percentile) or share of observed "
    "indicators.",
    "map": "Map: regions of the chosen level. Clicks alternate: the first sets the reference (✕), the second the "
    "comparison region (◆), the third a new reference, and so on; the bar above says what the next click does. "
    "Clicked regions are also added to the selection; a lasso or box (toolbar top right) replaces the selection.",
    "embedding": "Embedding: regions placed by profile similarity (nearby = similar). Orientation only; distances "
    "and types are computed in the full space. With indicator directions, an arrow points to where regions "
    "with high values of the indicator lie; the angle between two arrows approximates their correlation.",
    "neighbours": "Nearest regions to the reference under the current lens. Score = percentile among all pairs "
    "(100 = most similar); click a row to compare it with the reference.",
    "difference": "Difference profile: reference minus the comparison region per indicator (robust-scaled); "
    "filled = both observed, open = at least one predicted, bars = 90 % intervals.",
    "compare": "Region compared with the reference in the difference profile: the nearest region by default, a "
    "clicked table row, a region clicked on the map or in the embedding, or any region typed here.",
    "trajectories": "Movement of each region from the first to the last period window, in the same coordinate "
    "system as the embedding with indicator directions. Relative to the mean drift removes the common "
    "movement of all regions. The model of both windows is fitted on first use (1–2 minutes).",
}


def _embedding(Y: np.ndarray, method: str, seed: int = 0) -> tuple[np.ndarray, str]:
    from sklearn.manifold import trustworthiness

    from regsim.viz.views import embed_2d

    try:
        E, ev = embed_2d(Y, method, seed)
    except ImportError:
        E, ev = embed_2d(Y, "pca", seed)
        method = "pca"
    k = min(10, len(Y) - 2)
    trust = trustworthiness(Y, E, n_neighbors=k) if k >= 1 else float("nan")
    note = f"{method.upper()} of the similarity space: trustworthiness {trust:.2f}" + (
        f", explained variance {100 * ev:.0f} %" if ev else ""
    )
    return E, note


def pca_basis(profiles: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> dict:
    """PCA of the standardised ``profiles[rows][:, cols]``: centre, scale, two loadings and explained variance."""
    X = profiles[np.ix_(rows, cols)]
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    _, S, Vt = np.linalg.svd((X - mean) / sd, full_matrices=False)
    return {"mean": mean, "sd": sd, "loadings": Vt[:2].T, "ev": S**2 / np.sum(S**2)}


def project(basis: dict, X: np.ndarray) -> np.ndarray:
    """Scores of profiles ``X`` (same columns as the basis) in the basis."""
    return ((X - basis["mean"]) / basis["sd"]) @ basis["loadings"]


def arrows(basis: dict, scores: np.ndarray, names: list[str], pick: list[int]) -> list[dict]:
    """Indicator directions scaled into the range of ``scores``."""
    L = basis["loadings"]
    scale = 0.9 * min(np.abs(scores[:, d]).max() / max(np.abs(L[pick, d]).max(), 1e-12) for d in (0, 1))
    return [{"name": names[i], "x": float(L[i, 0] * scale), "y": float(L[i, 1] * scale)} for i in pick]


def create_dash(
    fitted: Fitted,
    geometry: dict | None = None,
    title: str = "regsim explorer",
    panel: Callable[[], Fitted] | None = None,
    region_names: dict[str, str] | None = None,
    indicator_labels: dict[str, str] | None = None,
    indicator_sources: dict[str, str] | None = None,
    n_arrows: int = 12,
    max_trajectories: int = 60,
):
    """Dash app around a fitted (calibrated) model.

    Parameters
    ----------
    fitted : Fitted
        Calibrated model of the latest period window.
    geometry : dict, optional
        Short geo code -> shapely polygon (:func:`regsim.viz.views.load_geometry`); without it the map is empty.
    panel : callable, optional
        Returns the model fitted on all period windows (geo-time entities); called on first use (trajectories,
        indicator directions).
    region_names, indicator_labels : dict, optional
        Geo id -> region name and indicator id -> label for display.
    indicator_sources : dict, optional
        Indicator id -> source connector (``eurostat``, ``d36``, ``ardeco``) for grouping and colours.
    """
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, percentile_score, s_profile
    from regsim.typology.run import TypologyRun, run_typology
    from regsim.uncertainty.domain import applicability
    from regsim.viz import views as V

    model, data = fitted.model, fitted.data
    base = fitted.base
    levels = sorted(int(x) for x in np.unique(data.levels()))
    domain = applicability(model, data).domain
    names = {g: str(n).strip() for g, n in (region_names or {}).items() if n and str(n).strip()}
    labels = indicator_labels or {}
    sources = {c: (indicator_sources or {}).get(c, "other") for c in data.column_ids}
    present = [s for s in SOURCES if s in set(sources.values())]
    load_panel = lru_cache(maxsize=1)(panel) if panel else None
    app = Dash(__name__, title=title)

    def name(g: str) -> str:
        """Region name with its code, e.g. 'Wien (AT130)'."""
        code = short_id(g)
        return f"{names[g.split('@')[0]]} ({code})" if g.split("@")[0] in names else code

    def short_name(g: str, n: int = 22) -> str:
        nm = names.get(g.split("@")[0], short_id(g))
        return nm if len(nm) <= n else nm[: n - 1] + "…"

    def caption(key):
        return html.Div(HELP[key], style=CAPTION)

    def short_label(c: str, n: int = 30) -> str:
        """Short readable indicator label, e.g. 'Transport: Road Density'."""
        lab = re.sub(r"\s*\([^)]{6,}\)", "", labels.get(c, c))  # long explanations stay in the hover text
        for a, b in SHORTEN.items():
            lab = lab.replace(a, b)
        lab = lab[:1].upper() + lab[1:]
        return lab if len(lab) <= n else lab[: n - 1] + "…"

    def full_label(c: str) -> str:
        return f"{labels.get(c, c)} ({c}) · {SOURCES.get(sources[c], sources[c])}"

    def relabel(text: str) -> str:
        """Replace indicator ids in a text (type labels, contributor lists) by short labels."""
        for c in sorted(data.column_ids, key=len, reverse=True):
            text = text.replace(c, short_label(c, 26))
        return text

    def legend():
        return html.Span(
            [html.Span("Arrow colours: ")]
            + [
                html.Span(f"■ {SOURCES[s]}  ", style={"color": SOURCE_COLOURS[s], "marginRight": "6px"})
                for s in present
            ],
            style=CAPTION,
        )

    level_options = [{"label": f"NUTS-{lv}" if lv < 4 else "LAU", "value": lv} for lv in levels]
    lens_options = sorted(
        (
            {"label": f"[{SOURCES.get(sources[c], sources[c])}] {labels.get(c, c)} ({c})", "value": c}
            for c in data.column_ids
        ),
        key=lambda o: o["label"].lower(),
    )
    source_buttons = html.Div(
        [html.Span("Lens by source: ", title=HELP["sources"], style={"fontSize": "13px"})]
        + [html.Button("all indicators", id="src-all", n_clicks=0, style={"marginRight": "6px"})]
        + [
            html.Button(
                f"{SOURCES[s]} only ({sum(v == s for v in sources.values())})",
                id=f"src-{s}",
                n_clicks=0,
                style={"marginRight": "6px", "borderColor": SOURCE_COLOURS[s]},
            )
            for s in present
        ],
        style={"margin": "6px 0"},
    )
    controls = html.Div(
        [
            html.Div(
                [html.Label("Level"), dcc.Dropdown(level_options, levels[-1], id="level", clearable=False)],
                style={"width": "110px"},
            ),
            html.Div(
                [
                    html.Label("Lens", title=HELP["lens"]),
                    dcc.Dropdown(lens_options, [], id="lens", multi=True, placeholder="all indicators"),
                ],
                style={"flex": 2, "minWidth": "300px"},
            ),
            html.Div(
                [
                    html.Label("Types", title=HELP["types"]),
                    dcc.Dropdown(
                        [{"label": "auto", "value": 0}]
                        + [{"label": str(k), "value": k} for k in range(2, 13)],
                        0,
                        id="k",
                        clearable=False,
                    ),
                ],
                style={"width": "90px"},
            ),
            html.Div(
                [
                    html.Label("Reference region"),
                    dcc.Dropdown([], None, id="reference", placeholder="click a region or type a name"),
                ],
                style={"flex": 1, "minWidth": "260px"},
            ),
            html.Div(
                [
                    html.Label("Compare with", title=HELP["compare"]),
                    dcc.Dropdown([], None, id="partner", placeholder="nearest region, click or type a name"),
                ],
                style={"flex": 1, "minWidth": "260px"},
            ),
            html.Div(
                [
                    html.Label("Colour", title=HELP["colour"]),
                    dcc.RadioItems(
                        ["type", "similarity", "observed share"], "type", id="colour", inline=True
                    ),
                ]
            ),
            html.Div(
                [html.Label("Embedding"), dcc.RadioItems(["pca", "umap"], "pca", id="method", inline=True)]
            ),
        ],
        style={"display": "flex", "gap": "14px", "alignItems": "flex-end", "flexWrap": "wrap"},
    )
    options = html.Div(
        [
            dcc.Checklist(
                [
                    {
                        "label": " show indicator directions (PCA, shared with the trajectories)",
                        "value": "show",
                    },
                    {"label": f" only lens indicators (else the {n_arrows} strongest)", "value": "lens"},
                ],
                [],
                id="arrows",
                inline=True,
                inputStyle={"marginLeft": "12px"},
            ),
        ],
        style={"margin": "6px 0"},
    )

    def hint(next_click, ref):
        """What the next click on the map or in the embedding does."""
        badge = {"background": "#fff3bf", "padding": "1px 6px", "borderRadius": "3px"}
        if next_click == "partner" and ref:
            what = [html.B("comparison region ◆"), f" (compared with the reference {name(ref)})"]
        else:
            what = [html.B("reference region ✕"), " (then the next click sets the comparison region)"]
        return html.Span(["Next click on the map or in the embedding sets the ", *what], style=badge)

    selection_box = html.Div(
        [
            html.Div(hint("reference", None), id="click-hint", style={"marginBottom": "2px"}),
            html.Span(id="selection-info"),
            html.Button("clear selection", id="clear", n_clicks=0, style={"marginLeft": "10px"}),
        ],
        style={"fontSize": "13px", "margin": "4px 0", "padding": "4px 8px", "background": "#f4f6f8"},
    )
    similar_tab = html.Div(
        [
            html.Div(
                [
                    caption("neighbours"),
                    dash_table.DataTable(
                        id="neighbours",
                        page_size=10,
                        cell_selectable=True,
                        style_table={"width": "660px"},
                        style_cell_conditional=[{"if": {"column_id": "region"}, "minWidth": "200px"}],
                        style_cell={
                            "fontSize": "12px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ]
            ),
            html.Div(
                [caption("difference"), dcc.Loading(dcc.Graph(id="difference"))],
                style={"flex": 1},
            ),
        ],
        style={"display": "flex", "gap": "16px", "marginTop": "8px"},
    )
    traj_tab = html.Div(
        [
            caption("trajectories"),
            dcc.Checklist(
                [
                    {
                        "label": " follow map/embedding selection (else the capital regions)",
                        "value": "follow",
                    },
                    {"label": " relative to the mean drift of all regions", "value": "relative"},
                    {"label": " show indicator directions", "value": "arrows"},
                ],
                [],
                id="traj-opts",
                inline=True,
                inputStyle={"marginLeft": "12px"},
                style={"margin": "4px 0 8px 0"},
            ),
            dcc.Loading(dcc.Graph(id="trajectories", style={"height": "660px"})),
            legend(),
        ],
        style={"marginTop": "8px"},
    )
    app.layout = html.Div(
        [
            html.H3(title, style={"marginBottom": "2px"}),
            html.Div(HELP["intro"], style={**CAPTION, "fontSize": "13px"}),
            controls,
            source_buttons,
            options,
            dcc.Loading(html.Div(id="summary", style={"fontSize": "13px", "margin": "4px 0"})),
            selection_box,
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="map", style={"height": "560px"}), caption("map")], style={"flex": 1}
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="embedding", style={"height": "560px"}),
                            caption("embedding"),
                            legend(),
                        ],
                        style={"flex": 1},
                    ),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="similar",
                children=[
                    dcc.Tab(
                        label="Nearest regions and difference profile",
                        value="similar",
                        children=[similar_tab],
                    ),
                    dcc.Tab(label="Trajectories", value="traj", children=[traj_tab]),
                ],
            ),
            dcc.Store(id="state"),
            dcc.Store(id="selection"),
            dcc.Store(id="click-next", data="reference"),
        ],
        style={"fontFamily": "sans-serif", "margin": "12px"},
    )

    def _lens(level, cols):
        return Lens(indicators=tuple(cols) if cols else None, levels=(level,))

    @lru_cache(maxsize=16)
    def _basis(level: int, cols: tuple[str, ...]):
        """Shared PCA basis from the panel model (both windows) for the lens and level, or None."""
        if load_panel is None:
            return None
        pf = load_panel()
        names_c = list(cols) if cols else [c for c in data.column_ids if c in pf.data.column_ids]
        pcols = np.array([pf.data.column_ids.index(c) for c in names_c])
        prow = np.flatnonzero(pf.data.levels() == level)
        b = pca_basis(pf.base.predict(), prow, pcols)
        b.update(names=names_c, panel_rows=prow, panel_cols=pcols)
        return b

    def _pick(b, scores, cols, choice):
        if "lens" in (choice or []) and cols:
            pick = [b["names"].index(c) for c in cols if c in b["names"]]
        else:
            L = b["loadings"]
            pick = list(np.argsort(-np.hypot(L[:, 0], L[:, 1]))[:n_arrows])
        return arrows(b, scores, b["names"], pick)

    def _draw_arrows(fig, directions):
        for a in directions:
            c = a["name"]
            V.add_arrow(
                fig,
                0.0,
                0.0,
                a["x"],
                a["y"],
                short_label(c),
                colour=SOURCE_COLOURS.get(sources[c], "#444"),
                hover=full_label(c),
            )

    if present:

        @app.callback(
            Output("lens", "value"),
            Input("src-all", "n_clicks"),
            *[Input(f"src-{s}", "n_clicks") for s in present],
            prevent_initial_call=True,
        )
        def lens_by_source(*_clicks):
            src = (ctx.triggered_id or "src-all").removeprefix("src-")
            return [] if src == "all" else [c for c in data.column_ids if sources[c] == src]

    @app.callback(
        Output("state", "data"),
        Output("summary", "children"),
        Output("reference", "options"),
        Output("partner", "options"),
        Input("level", "value"),
        Input("lens", "value"),
        Input("k", "value"),
        Input("method", "value"),
        Input("arrows", "value"),
    )
    def compute(level, cols, k, method, choice):
        lens = _lens(level, cols)
        space = s_profile(model, data, lens)
        rows = space.lens.rows
        ty = run_typology(base, data, TypologyRun(lens=lens, k=k or None, k_range=(2, 10), n_boot=10))
        directions = []
        choice = choice or []
        if "show" in choice and method == "pca":
            b = _basis(level, tuple(cols or ()))
            if b is not None:
                cur_cols = np.array([data.column_ids.index(c) for c in b["names"]])
                E = project(b, base.predict()[np.ix_(rows, cur_cols)])
                ref_scores = project(b, load_panel().base.predict()[np.ix_(b["panel_rows"], b["panel_cols"])])
                directions = _pick(b, ref_scores, cols, choice)
                ev = b["ev"]
                note = (
                    f"PCA of the standardised profiles of both period windows (shared with the trajectories): "
                    f"PC 1 {100 * ev[0]:.0f} %, PC 2 {100 * ev[1]:.0f} % of variance"
                )
            else:
                _, E, L, bnames, ev = V.biplot_coords(model, data, lens, level)
                pick = (
                    [bnames.index(c) for c in cols if c in bnames]
                    if "lens" in choice and cols
                    else list(np.argsort(-np.hypot(L[:, 0], L[:, 1]))[:n_arrows])
                )
                directions = [{"name": bnames[i], "x": float(L[i, 0]), "y": float(L[i, 1])} for i in pick]
                note = f"PCA of the standardised profiles: PC 1 {100 * ev[0]:.0f} %, PC 2 {100 * ev[1]:.0f} %"
        else:
            E, note = _embedding(space.Y[rows], method)
        ids = [data.geo_ids[i] for i in rows]
        state = {
            "directions": directions,
            "ids": ids,
            "x": E[:, 0].tolist(),
            "y": E[:, 1].tolist(),
            "type": ty.labels.tolist(),
            "share": data.mask[rows][:, space.lens.cols].mean(axis=1).tolist(),
            "labels": ty.cluster_table.label.to_dict(),
        }
        in_lens = cols or data.column_ids
        by_source = ", ".join(
            f"{SOURCES[s]} {sum(sources[c] == s for c in in_lens)}"
            for s in present
            if any(sources[c] == s for c in in_lens)
        )
        summary = (
            f"{V.lens_label(data, lens, level)} ({by_source}) · {ty.k} types (bootstrap stability ARI {ty.stability:.2f}) · "
            f"embedding: {note}"
        )
        options = sorted(({"label": name(g), "value": g} for g in ids), key=lambda o: o["label"].lower())
        return state, summary, options, options

    @app.callback(
        Output("reference", "value"),
        Output("partner", "value"),
        Output("click-next", "data"),
        Input("map", "clickData"),
        Input("embedding", "clickData"),
        Input("neighbours", "active_cell"),
        Input("reference", "value"),
        Input("partner", "value"),
        State("click-next", "data"),
        State("lens", "value"),
    )
    def pick(map_click, emb_click, cell, ref, partner, next_click, cols):
        """Clicks on the map or in the embedding alternate between the reference and the comparison region; a
        new reference starts with its nearest region as comparison region."""
        fired = set(ctx.triggered_prop_ids)
        for prop, click in (("map.clickData", map_click), ("embedding.clickData", emb_click)):
            g = ((click or {}).get("points") or [{}])[0].get("customdata")
            if prop in fired and g:
                if next_click == "partner" and ref and g != ref:
                    return no_update, g, "reference"
                return g, _nearest(g, cols), "partner"
        if "neighbours.active_cell" in fired and cell and cell.get("row_id"):
            return no_update, cell["row_id"], "reference"
        if "reference.value" in fired:  # typed reference: the next click picks the comparison region
            return no_update, _nearest(ref, cols), "partner" if ref else "reference"
        if "partner.value" in fired:
            return no_update, no_update, "reference"
        return no_update, no_update, no_update

    def _nearest(ref, cols) -> str | None:
        """Nearest region to ``ref`` under the lens (default comparison region)."""
        if not ref or ref not in data.geo_ids:
            return None
        level = int(data.levels()[data.geo_ids.index(ref)])
        nb = nearest(s_profile(model, data, _lens(level, cols)), ref, k=1, n_contrib=0)
        return nb.geo_id.iloc[0] if len(nb) else None

    @app.callback(
        Output("click-hint", "children"),
        Input("click-next", "data"),
        Input("reference", "value"),
    )
    def click_hint(next_click, ref):
        return hint(next_click, ref)

    @app.callback(
        Output("selection", "data"),
        Input("map", "selectedData"),
        Input("embedding", "selectedData"),
        Input("map", "clickData"),
        Input("embedding", "clickData"),
        Input("clear", "n_clicks"),
        Input("state", "data"),
        State("selection", "data"),
    )
    def select(sel_map, sel_emb, click_map, click_emb, _clear, state, current):
        """A lasso or box replaces the selection, a click adds the region, the button empties it. Empty
        selection events (sent by Plotly when a figure is redrawn or a click lands in lasso mode) are ignored."""
        fired = set(ctx.triggered_prop_ids)
        current = list(current or [])
        if "clear.n_clicks" in fired:
            return []
        for prop, sel in (("map.selectedData", sel_map), ("embedding.selectedData", sel_emb)):
            picked = {p.get("customdata") for p in (sel or {}).get("points", []) if p.get("customdata")}
            if prop in fired and picked:
                return sorted(picked)
        for prop, click in (("map.clickData", click_map), ("embedding.clickData", click_emb)):
            g = ((click or {}).get("points") or [{}])[0].get("customdata")
            if prop in fired and g:
                return sorted({*current, g})
        if "state.data" in fired:  # new level or lens: keep the regions still shown
            shown = set((state or {}).get("ids", []))
            return [g for g in current if g in shown]
        return no_update

    @app.callback(
        Output("selection-info", "children"),
        Input("selection", "data"),
        Input("reference", "value"),
        Input("partner", "value"),
    )
    def selection_info(selection, ref, partner):
        lead = [
            html.B("Reference ✕: "),
            name(ref) if ref else "none",
            html.B(" · compared with ◆: "),
            name(partner) if partner else "none",
            html.Br(),
            html.B("Selected regions: "),
        ]
        if not selection:
            return [*lead, "none (click regions, or draw a lasso or box on the map or in the embedding)"]
        shown = ", ".join(name(g) for g in selection[:25])
        more = f" … and {len(selection) - 25} more" if len(selection) > 25 else ""
        return [*lead, f"{len(selection)}: {shown}{more}"]

    def _values(state, colour, ref):
        if colour == "type":
            return pd.Series(state["type"], index=state["ids"])
        if colour == "observed share":
            return pd.Series(state["share"], index=state["ids"], dtype=float)
        if not ref or ref not in data.geo_ids:
            return pd.Series(np.nan, index=state["ids"], dtype=float)
        level = int(data.levels()[data.geo_ids.index(state["ids"][0])])
        space = s_profile(model, data, Lens(levels=(level,)))
        rows = np.array([data.geo_ids.index(g) for g in state["ids"]])
        d = space.distances(data.geo_ids.index(ref), rows)
        return pd.Series(percentile_score(d, space.pair_distribution(level)), index=state["ids"], dtype=float)

    def _outline(codes) -> tuple[list, list]:
        """Boundary lines of the regions ``codes`` (None-separated, one trace for all)."""
        xs: list = []
        ys: list = []
        for c in codes:
            geom = geometry.get(c)
            for poly in getattr(geom, "geoms", [geom] if geom is not None else []):
                if not hasattr(poly, "exterior"):
                    continue
                x, y = poly.exterior.xy
                xs += [*x, None]
                ys += [*y, None]
        return xs, ys

    @app.callback(
        Output("map", "figure"),
        Output("embedding", "figure"),
        Input("state", "data"),
        Input("colour", "value"),
        Input("reference", "value"),
        Input("partner", "value"),
        Input("selection", "data"),
    )
    def figures(state, colour, ref, partner, selection):
        if not state:
            return go.Figure(), go.Figure()
        values = _values(state, colour, ref)
        chosen = set(selection or [])
        ids = state["ids"]
        idx = [i for i, g in enumerate(ids) if g in chosen] or None
        rev = (
            f"{len(ids)}:{ids[0]}:{state['x'][0]:.6g}"  # keeps zoom and pan while the embedding is unchanged
        )
        if colour == "type":
            marker = {"size": 6, "color": [V._hex(c) for c in V._colours(values.to_numpy())]}
        else:
            bar = "similarity (percentile)" if colour == "similarity" else "observed share"
            marker = {
                "size": 6,
                "color": values.to_numpy(),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": bar},
            }
        hover = [
            f"{name(g)}<br>type {t}: {relabel(state['labels'].get(str(t), ''))}<br>applicability: {domain.get(g, '')}"
            for g, t in zip(ids, state["type"], strict=True)
        ]
        emb = go.Figure(
            go.Scattergl(
                x=state["x"],
                y=state["y"],
                mode="markers",
                marker=marker,
                customdata=ids,
                text=hover,
                hoverinfo="text",
                selectedpoints=idx,
                unselected={"marker": {"opacity": 0.25}},
            )
        )
        marks = [(ref, "x", "black"), (partner if partner != ref else None, "diamond", PARTNER_COLOUR)]
        for g, symbol, colour_ in marks:
            if g in ids:
                i = ids.index(g)
                emb.add_trace(
                    go.Scatter(
                        x=[state["x"][i]],
                        y=[state["y"][i]],
                        mode="markers+text",
                        text=[short_name(g)],
                        textposition="top center",
                        textfont={"color": colour_},
                        marker={"size": 14, "symbol": symbol, "color": colour_},
                        hoverinfo="skip",
                    )
                )
        _draw_arrows(emb, state.get("directions", []))
        emb.update_layout(
            dragmode="lasso",
            uirevision=rev,
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title="Embedding" + (" with indicator directions" if state.get("directions") else ""),
        )
        if geometry:
            short_names = {short_id(g): names[g] for g in ids if g in names}
            mp = V.maps({"": values.rename(index=short_id)}, geometry, interactive=True, names=short_names)
            full = {short_id(g): g for g in ids}
            for tr in mp.data:
                if tr.customdata is not None:  # centroid markers: carry full ids for the linking
                    tr.customdata = [full.get(c, c) for c in tr.customdata]
                    if chosen:
                        tr.selectedpoints = [j for j, c in enumerate(tr.customdata) if c in chosen]
                        tr.selected = {"marker": {"color": "red", "size": 7, "opacity": 1}}
            outline = _outline(short_id(g) for g in chosen)
            if outline[0]:
                mp.add_trace(
                    go.Scatter(
                        x=outline[0],
                        y=outline[1],
                        mode="lines",
                        line={"color": "red", "width": 1},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            for g, symbol, colour_ in marks:
                if g and short_id(g) in geometry:
                    x, y = geometry[short_id(g)].representative_point().coords[0]
                    mp.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            text=[short_name(g)],
                            textposition="top center",
                            textfont={"color": colour_},
                            marker={"size": 12, "symbol": symbol, "color": colour_},
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            mp.update_layout(title="Map", dragmode="lasso", uirevision=rev)
        else:
            mp = go.Figure().update_layout(title="no geometry configured")
        return mp, emb

    @app.callback(
        Output("neighbours", "data"),
        Output("neighbours", "columns"),
        Input("reference", "value"),
        Input("state", "data"),
        State("lens", "value"),
    )
    def neighbours(ref, state, cols):
        if not ref or not state or ref not in data.geo_ids:
            return [], []
        level = int(data.levels()[data.geo_ids.index(ref)])
        nb = nearest(s_profile(model, data, _lens(level, cols)), ref, k=10)
        table = pd.DataFrame(
            {
                "region": [name(g) for g in nb.geo_id],
                "score": nb.score.round(1),
                "distance": nb.distance.round(2),
                "observed share": nb.observed_share.round(2),
                "differs most in": [relabel(x) for x in nb.top_contributors],
                "id": nb.geo_id,
            }
        )
        columns = [{"name": c, "id": c} for c in table.columns if c != "id"]
        return table.to_dict("records"), columns

    @app.callback(
        Output("neighbours", "style_data_conditional"),
        Input("partner", "value"),
        Input("neighbours", "data"),
    )
    def mark_partner(partner, rows):
        return [
            {"if": {"row_index": i}, "backgroundColor": "#fde2e2", "fontWeight": "bold"}
            for i, r in enumerate(rows or [])
            if r["id"] == partner
        ]

    def _pair(ref: str, partner: str, cols) -> str:
        """Overall similarity of the pair under the lens: percentile score, distance and rank of ``partner``
        among all regions of the reference's level."""
        g, h = data.geo_ids.index(ref), data.geo_ids.index(partner)
        level = int(data.levels()[g])
        space = s_profile(model, data, _lens(level, cols))
        rows = space.lens.rows
        rows = rows[(space.level[rows] == level) & (rows != g)]
        d = space.distances(g, rows)
        d_pair = float(space.distances(g, np.array([h]))[0])
        score = float(percentile_score(np.array([d_pair]), space.pair_distribution(level))[0])
        rank = f"rank {int((d < d_pair).sum()) + 1} of {rows.size}" if h in rows else "other level"
        return f"score {score:.0f} (percentile, 100 = most similar) · distance {d_pair:.2f} · {rank}"

    @app.callback(
        Output("difference", "figure"),
        Input("reference", "value"),
        Input("partner", "value"),
        State("lens", "value"),
    )
    def difference(ref, partner, cols):
        if not ref or not partner or ref == partner or {ref, partner} - set(data.geo_ids):
            return go.Figure().update_layout(
                title="choose a reference and a comparison region (click, table or search) to compare"
            )
        fig = V.difference_profile(
            model,
            data,
            ref,
            partner,
            lens=Lens(indicators=tuple(cols) if cols else None),
            interactive=True,
            labels={c: short_label(c, 40) for c in data.column_ids},
            hover={c: full_label(c) for c in data.column_ids},
        )
        sub = fig.layout.title.text.split("<br>", 1)
        fig.update_layout(
            title={
                "text": f"Difference profile: {name(ref)} − {name(partner)}<br><sup>{_pair(ref, partner, cols)}"
                + (f"<br>{sub[1]}" if len(sub) > 1 else "")
                + "</sup>",
                "font": {"size": 14},
            },
            margin={"t": 90},
        )
        return fig

    @app.callback(
        Output("trajectories", "figure"),
        Input("tabs", "value"),
        Input("traj-opts", "value"),
        Input("selection", "data"),
        Input("level", "value"),
        Input("lens", "value"),
        State("arrows", "value"),
    )
    def trajectories(tab, opts, selection, level, cols, choice):
        if tab != "traj":
            return no_update
        if load_panel is None:
            return go.Figure().update_layout(
                title="trajectories need at least two period windows in the config"
            )
        opts = opts or []
        pf = load_panel()
        b = _basis(level, tuple(cols or ()))
        P = pf.base.predict()[np.ix_(b["panel_rows"], b["panel_cols"])]
        scores = project(b, P)
        pids = [pf.data.geo_ids[i] for i in b["panel_rows"]]
        base_ids = [g.split("@")[0] for g in pids]
        periods = [g.split("@")[1] for g in pids]
        first, last = min(periods), max(periods)
        pos = {(g, p): s for g, p, s in zip(base_ids, periods, scores, strict=True)}
        disp = {
            g: pos[(g, last)] - pos[(g, first)]
            for g in set(base_ids)
            if (g, first) in pos and (g, last) in pos
        }
        drift = np.mean(list(disp.values()), axis=0) if "relative" in opts else np.zeros(2)
        if "follow" in opts:
            chosen = [g for g in (selection or []) if g in disp]
            if not chosen:
                return go.Figure().update_layout(
                    title="select regions on the map or in the embedding (lasso or box)"
                )
            note = f"{len(chosen)} selected regions"
        else:
            chosen = sorted(g for g in disp if any(short_id(g).startswith(c) for c in CAPITAL_REGIONS))
            note = "capital regions"
        if len(chosen) > max_trajectories:
            note += f" (first {max_trajectories} of {len(chosen)})"
            chosen = chosen[:max_trajectories]
        med = float(np.median([np.linalg.norm(v - drift) for v in disp.values()]))
        ev = b["ev"]
        fig = go.Figure(
            go.Scattergl(
                x=scores[:, 0],
                y=scores[:, 1],
                mode="markers",
                marker={"size": 3, "color": "#d9d9d9"},
                text=[f"{name(g)} {p}" for g, p in zip(base_ids, periods, strict=True)],
                hoverinfo="text",
                showlegend=False,
            )
        )
        for g in chosen:
            p0 = pos[(g, first)]
            p1 = p0 + disp[g] - drift
            fig.add_trace(
                go.Scatter(
                    x=[p0[0], p1[0]],
                    y=[p0[1], p1[1]],
                    mode="markers",
                    marker={"size": [5, 1], "color": "#9ecae1"},
                    text=[f"{name(g)} {first}", f"{name(g)} {last}"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            V.add_arrow(fig, p0[0], p0[1], p1[0], p1[1], short_name(g), colour="#08519c")
        if "arrows" in opts:
            _draw_arrows(fig, _pick(b, scores, cols, choice))
        fig.update_layout(
            title=(
                f"Trajectories {first} → {last} ({note}; median displacement {med:.2f}"
                f"{', relative to mean drift' if 'relative' in opts else ''})"
            ),
            xaxis_title=f"PC 1 ({100 * ev[0]:.0f} %)",
            yaxis_title=f"PC 2 ({100 * ev[1]:.0f} %)",
            margin={"t": 50},
        )
        return fig

    return app
