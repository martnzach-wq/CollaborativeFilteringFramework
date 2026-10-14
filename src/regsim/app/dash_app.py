"""Interactive explorer (decision D7: Dash) with a linked map and embedding.

Selecting regions (lasso or box) on the map highlights them in the embedding and vice versa; clicking a region
makes it the reference, which shows its nearest regions and the difference profile (V6). Lens, level, number of
types and colouring are controls. Distances and types are computed in the full space; the embedding states its
explained variance (PCA) or trustworthiness (UMAP).

Run with ``regsim app CONFIG`` or ``create_dash(fitted, geometry).run()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regsim.api.runtime import Fitted, short_id


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
    note = f"{method.upper()}: trustworthiness {trust:.2f}" + (
        f", explained variance {100 * ev:.0f} %" if ev else ""
    )
    return E, note


def create_dash(fitted: Fitted, geometry: dict | None = None, title: str = "regsim explorer"):
    """Dash app around a fitted (calibrated) model. ``geometry`` maps short geo codes to shapely polygons
    (:func:`regsim.viz.views.load_geometry`); without it only the embedding is linked."""
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dash_table, dcc, html

    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, s_profile
    from regsim.typology.run import TypologyRun, run_typology
    from regsim.uncertainty.domain import applicability
    from regsim.viz import views as V

    model, data = fitted.model, fitted.data
    base = fitted.base
    levels = sorted(int(x) for x in np.unique(data.levels()))
    domain = applicability(model, data).domain
    app = Dash(__name__, title=title)

    app.layout = html.Div(
        [
            html.H3(title),
            html.Div(
                [
                    html.Label("Level"),
                    dcc.Dropdown(levels, levels[-1], id="level", clearable=False, style={"width": "90px"}),
                    html.Label("Lens"),
                    dcc.Dropdown(
                        data.column_ids,
                        [],
                        id="lens",
                        multi=True,
                        placeholder="all indicators",
                        style={"minWidth": "320px"},
                    ),
                    html.Label("Types"),
                    dcc.Dropdown(
                        [{"label": "auto", "value": 0}]
                        + [{"label": str(k), "value": k} for k in range(2, 13)],
                        0,
                        id="k",
                        clearable=False,
                        style={"width": "90px"},
                    ),
                    html.Label("Colour"),
                    dcc.RadioItems(
                        ["type", "similarity", "observed share"], "type", id="colour", inline=True
                    ),
                    html.Label("Embedding"),
                    dcc.RadioItems(["pca", "umap"], "pca", id="method", inline=True),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
            ),
            html.Div(id="summary", style={"margin": "6px 0", "fontSize": "13px"}),
            html.Div(
                [
                    dcc.Graph(id="map", style={"flex": 1, "height": "560px"}),
                    dcc.Graph(id="embedding", style={"flex": 1, "height": "560px"}),
                ],
                style={"display": "flex"},
            ),
            html.Div(
                [
                    dash_table.DataTable(
                        id="neighbours",
                        page_size=10,
                        style_table={"width": "560px"},
                        style_cell={
                            "fontSize": "12px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                    dcc.Graph(id="difference", style={"flex": 1, "height": "600px"}),
                ],
                style={"display": "flex", "gap": "12px"},
            ),
            dcc.Store(id="state"),
            dcc.Store(id="reference"),
        ],
        style={"fontFamily": "sans-serif", "margin": "12px"},
    )

    def _lens(level, cols):
        return Lens(indicators=tuple(cols) if cols else None, levels=(level,))

    @app.callback(
        Output("state", "data"),
        Output("summary", "children"),
        Input("level", "value"),
        Input("lens", "value"),
        Input("k", "value"),
        Input("method", "value"),
    )
    def compute(level, cols, k, method):
        lens = _lens(level, cols)
        space = s_profile(model, data, lens)
        rows = space.lens.rows
        ty = run_typology(base, data, TypologyRun(lens=lens, k=k or None, k_range=(2, 10), n_boot=10))
        E, note = _embedding(space.Y[rows], method)
        state = {
            "ids": [data.geo_ids[i] for i in rows],
            "x": E[:, 0].tolist(),
            "y": E[:, 1].tolist(),
            "type": ty.labels.tolist(),
            "share": data.mask[rows][:, space.lens.cols].mean(axis=1).tolist(),
            "labels": ty.cluster_table.label.to_dict(),
        }
        summary = (
            f"{V.lens_label(data, lens, level)} · {ty.k} types (bootstrap ARI {ty.stability:.2f}, "
            f"hash {ty.config_hash}) · {note}"
        )
        return state, summary

    @app.callback(
        Output("reference", "data"),
        Input("map", "clickData"),
        Input("embedding", "clickData"),
        State("reference", "data"),
    )
    def pick(map_click, emb_click, current):
        from dash import ctx

        click = map_click if ctx.triggered_id == "map" else emb_click
        if not click:
            return current
        return click["points"][0].get("customdata", current)

    def _values(state, colour, ref):
        if colour == "type":
            return pd.Series(state["type"], index=state["ids"])
        if colour == "observed share":
            return pd.Series(state["share"], index=state["ids"], dtype=float)
        if not ref or ref not in data.geo_ids:
            return pd.Series(np.nan, index=state["ids"], dtype=float)
        from regsim.similarity.entities import percentile_score

        level = int(data.levels()[data.geo_ids.index(state["ids"][0])])
        space = s_profile(model, data, Lens(levels=(level,)))
        rows = np.array([data.geo_ids.index(g) for g in state["ids"]])
        d = space.distances(data.geo_ids.index(ref), rows)
        return pd.Series(percentile_score(d, space.pair_distribution(level)), index=state["ids"], dtype=float)

    @app.callback(
        Output("map", "figure"),
        Output("embedding", "figure"),
        Input("state", "data"),
        Input("colour", "value"),
        Input("reference", "data"),
        Input("map", "selectedData"),
        Input("embedding", "selectedData"),
    )
    def figures(state, colour, ref, sel_map, sel_emb):
        from dash import ctx

        if not state:
            return go.Figure(), go.Figure()
        values = _values(state, colour, ref)
        sel = sel_map if ctx.triggered_id == "map" else sel_emb if ctx.triggered_id == "embedding" else None
        chosen = {p.get("customdata") for p in (sel or {}).get("points", [])}
        ids = state["ids"]
        idx = [i for i, g in enumerate(ids) if g in chosen] or None
        if colour == "type":
            colours = [V._hex(c) for c in V._colours(values.to_numpy())]
            marker = {"size": 6, "color": colours}
        else:
            marker = {"size": 6, "color": values.to_numpy(), "colorscale": "Viridis", "showscale": True}
        hover = [
            f"{short_id(g)} · type {t}: {state['labels'].get(str(t), '')} · {domain.get(g, '')}"
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
            )
        )
        if ref in ids:
            i = ids.index(ref)
            emb.add_trace(
                go.Scatter(
                    x=[state["x"][i]],
                    y=[state["y"][i]],
                    mode="markers",
                    showlegend=False,
                    marker={"size": 14, "symbol": "x", "color": "black"},
                    hoverinfo="skip",
                )
            )
        emb.update_layout(
            dragmode="lasso",
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title="embedding (orientation only; distances in the full space)",
        )
        if geometry:
            mp = V.maps({"": values.rename(index=short_id)}, geometry, interactive=True)
            full = {short_id(g): g for g in ids}
            for tr in mp.data:
                if tr.customdata is not None:  # centroid markers: carry full ids for the linking
                    tr.customdata = [full.get(c, c) for c in tr.customdata]
                    if chosen:
                        tr.selectedpoints = [j for j, c in enumerate(tr.customdata) if c in chosen]
                        tr.selected = {"marker": {"color": "red", "size": 7, "opacity": 1}}
            mp.update_layout(title="map", dragmode="lasso")
        else:
            mp = go.Figure().update_layout(title="no geometry configured")
        return mp, emb

    @app.callback(
        Output("neighbours", "data"),
        Output("neighbours", "columns"),
        Output("difference", "figure"),
        Input("reference", "data"),
        Input("state", "data"),
        State("lens", "value"),
    )
    def details(ref, state, cols):
        if not ref or not state or ref not in data.geo_ids:
            return [], [], go.Figure().update_layout(title="click a region to choose the reference")
        level = int(data.levels()[data.geo_ids.index(ref)])
        lens = _lens(level, cols)
        nb = nearest(s_profile(model, data, lens), ref, k=10)
        nb["geo_id"] = nb.geo_id.map(short_id)
        nb = nb.drop(columns="level").round(3)
        partner = data.geo_ids[[short_id(g) for g in data.geo_ids].index(nb.geo_id.iloc[0])]
        fig = V.difference_profile(
            model, data, ref, partner, lens=Lens(indicators=lens.indicators), interactive=True
        )
        return nb.to_dict("records"), [{"name": c, "id": c} for c in nb.columns], fig

    return app
