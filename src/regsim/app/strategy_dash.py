"""Strategy dashboard (merge step M5, decision D66): levers of a local strategy as multiples of its stated level,
live model runs, targets and margins, response and climate bands kept apart, lever contributions, scenario
comparison and the assumptions behind each lever. City-independent; a city supplies a `StrategyAdapter`.

Run with ``create_strategy_app(adapter).run()``; ``app.server`` is the WSGI application for a server.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go

from regsim.app.strategy_adapter import (
    SeriesSpec,
    StrategyAdapter,
    TargetSpec,
    contributions,
    display,
    target_status,
    tr,
)

# reference palette (dataviz skill): categorical slots in fixed order, status colours reserved
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
C_SCEN, C_REF, C_CLIM = SERIES[0], "#8a8984", SERIES[1]
BAND_RESP, BAND_CLIM = "rgba(37,125,224,0.16)", "rgba(244,114,48,0.18)"
GOOD, CRITICAL = "#0aac23", "#dc3c3d"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MILESTONES = (2019, 2025, 2030, 2035, 2040, 2050)

TEXT = {
    "language": {"de": "Sprache", "en": "Language"},
    "preset": {"de": "Szenario laden", "en": "Load a scenario"},
    "bands": {"de": "Bandbreiten", "en": "Bands"},
    "band_response": {"de": "Reaktionsband (schwach / stark)", "en": "Response band (weak / strong)"},
    "band_climate": {
        "de": "Klimaband (alle Klimamodelle und Szenarien)",
        "en": "Climate band (all climate members)",
    },
    "levers": {
        "de": "Hebel (Vielfache m des Strategie-Niveaus)",
        "en": "Levers (multiples m of the strategy level)",
    },
    "options": {"de": "Annahmen", "en": "Assumptions"},
    "tab_overview": {"de": "Übersicht", "en": "Overview"},
    "tab_targets": {"de": "Ziele und Reserven", "en": "Targets and margins"},
    "tab_bands": {"de": "Bandbreiten", "en": "Bands"},
    "tab_contrib": {"de": "Beiträge der Hebel", "en": "Lever contributions"},
    "tab_compare": {"de": "Vergleich", "en": "Compare"},
    "tab_gaps": {"de": "Annahmen und Datenlücken", "en": "Assumptions and data gaps"},
    "met": {"de": "erreicht", "en": "met"},
    "missed": {"de": "verfehlt", "en": "missed"},
    "hard": {"de": "hart", "en": "hard"},
    "reported": {"de": "berichtet", "en": "reported"},
    "central": {"de": "zentral", "en": "central"},
    "target": {"de": "Ziel", "en": "Target"},
    "model": {"de": "Modell", "en": "Model"},
    "margin": {"de": "Reserve", "en": "Margin"},
    "year": {"de": "Jahr", "en": "Year"},
    "climate_min": {"de": "Klima, ungünstigstes Mitglied", "en": "climate, worst member"},
    "compute": {"de": "Beiträge berechnen", "en": "Compute contributions"},
    "contrib_note": {
        "de": "Änderung des Zielwerts, wenn der Hebel auf seine Referenz (m = 0) gesetzt wird; positiv: der Hebel "
        "hilft dem Ziel. Ein Modelllauf je Hebel.",
        "en": "Change of the target value when the lever is set to its reference (m = 0); positive: the lever "
        "helps the target. One model run per lever.",
    },
    "compare_pick": {"de": "Szenarien zum Vergleich", "en": "Scenarios to compare"},
    "current": {"de": "aktuelle Einstellung", "en": "current setting"},
    "effect_modelled": {"de": "modelliert", "en": "modelled"},
    "effect_stated effect": {"de": "gesetzte Wirkung", "en": "stated effect"},
    "effect_exogenous path": {"de": "exogener Pfad", "en": "exogenous path"},
    "conf_stated": {"de": "festgelegt", "en": "stated"},
    "conf_derived": {"de": "abgeleitet", "en": "derived"},
    "conf_assumed": {"de": "angenommen", "en": "assumed"},
    "gaps": {"de": "Datenlücken", "en": "data gaps"},
    "band_legend_resp": {"de": "Reaktionsband", "en": "response band"},
    "band_legend_clim": {"de": "Klimaband (min–max)", "en": "climate band (min–max)"},
    "climate_median": {"de": "Median der Klimamitglieder", "en": "median of the climate members"},
    "no_climate": {
        "de": "Für diese Stadt ist kein Klimaband hinterlegt.",
        "en": "No climate band for this city.",
    },
}
EFFECT_COLOUR = {"modelled": SERIES[0], "stated effect": SERIES[6], "exogenous path": "#8a8984"}


def t(key: str, lang: str) -> str:
    return tr(TEXT.get(key, key), lang)


def _layout(fig: go.Figure, title: str, unit: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": INK}, "x": 0, "xanchor": "left"},
        font={"family": FONT, "color": INK2, "size": 12},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 56, "r": 16, "t": 40, "b": 36},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        height=300,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=unit, title_font={"size": 11})
    return fig


# ---- figures (pure functions) -------------------------------------------------------------------------------


def fig_series(
    spec: SeriesSpec,
    central: pd.DataFrame,
    lang: str,
    bands: dict[str, pd.DataFrame] | None = None,
    references: dict[str, dict] | None = None,
    targets: list[TargetSpec] | None = None,
) -> go.Figure:
    """A pathway: central line, response band (envelope of the bands), reference lines and target markers."""
    fig = go.Figure()
    y = display(central, spec)
    if bands:
        env = pd.concat([display(r, spec) for r in bands.values()] + [y], axis=1)
        lo, hi = env.min(axis=1), env.max(axis=1)
        fig.add_trace(
            go.Scatter(x=list(hi.index), y=hi.values, line={"width": 0}, hoverinfo="skip", showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=list(lo.index),
                y=lo.values,
                line={"width": 0},
                fill="tonexty",
                fillcolor=BAND_RESP,
                name=t("band_legend_resp", lang),
                hoverinfo="skip",
            )
        )
    for ref in (references or {}).values():
        res = ref["results"]
        if spec.quantity in res.columns:
            r = display(res, spec).dropna()
            fig.add_trace(
                go.Scatter(
                    x=list(r.index),
                    y=r.values,
                    name=tr(ref["label"], lang),
                    line={"color": C_REF, "width": 2, "dash": "dash"},
                )
            )
    fig.add_trace(
        go.Scatter(x=list(y.index), y=y.values, name=t("model", lang), line={"color": C_SCEN, "width": 2})
    )
    pts = [tg for tg in (targets or []) if tg.quantity == spec.quantity]
    if pts:
        ty = [100 * tg.value if spec.relative_to is not None else tg.value * spec.scale for tg in pts]
        fig.add_trace(
            go.Scatter(
                x=[tg.year for tg in pts],
                y=ty,
                mode="markers",
                name=t("target", lang),
                marker={
                    "symbol": ["diamond" if tg.hard else "diamond-open" for tg in pts],
                    "size": 10,
                    "color": INK,
                    "line": {"color": "#ffffff", "width": 2},
                },
                text=[t("hard" if tg.hard else "reported", lang) for tg in pts],
                hovertemplate="%{y:.1f} (%{text})<extra>" + t("target", lang) + "</extra>",
            )
        )
    return _layout(fig, tr(spec.label, lang), tr(spec.unit, lang))


def fig_stack(
    parts: list[SeriesSpec], central: pd.DataFrame, title: str, lang: str, unit: str = ""
) -> go.Figure:
    """Composition at the milestone years: stacked bars, categorical slots in fixed order, 2px surface gaps."""
    years = [y for y in MILESTONES if y in central.index]
    fig = go.Figure()
    for i, p in enumerate(parts):
        v = display(central, p).loc[years]
        fig.add_trace(
            go.Bar(
                x=[str(y) for y in years],
                y=v.values,
                name=tr(p.label, lang),
                marker={"color": SERIES[i % len(SERIES)], "line": {"color": "#ffffff", "width": 2}},
            )
        )
    fig.update_layout(barmode="stack", bargap=0.45)
    return _layout(fig, title, unit)


def fig_climate(spec: SeriesSpec, climate: pd.DataFrame, central: pd.DataFrame, lang: str) -> go.Figure:
    """Climate band of one series: min-max over the members, their median, and the selected member."""
    sub = climate[climate.quantity == spec.quantity].pivot_table(
        index="year", columns="member", values="value"
    )
    sub = sub * spec.scale
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sub.index), y=sub.max(axis=1).values, line={"width": 0}, hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sub.index),
            y=sub.min(axis=1).values,
            line={"width": 0},
            fill="tonexty",
            fillcolor=BAND_CLIM,
            name=t("band_legend_clim", lang),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sub.index),
            y=sub.median(axis=1).values,
            name=t("climate_median", lang),
            line={"color": C_CLIM, "width": 2, "dash": "dot"},
        )
    )
    y = display(central, spec).dropna()
    fig.add_trace(
        go.Scatter(x=list(y.index), y=y.values, name=t("model", lang), line={"color": C_SCEN, "width": 2})
    )
    return _layout(fig, tr(spec.label, lang), tr(spec.unit, lang))


def fig_contributions(
    df: pd.DataFrame, quantity: str, year: int, labels: dict[str, str], title: str
) -> go.Figure:
    """Ranked horizontal bars of the lever contributions to one target."""
    sub = df[(df.quantity == quantity) & (df.year == year)].copy()
    sub = sub[sub.contribution.abs() > 1e-9].sort_values("contribution")
    fig = go.Figure(
        go.Bar(
            x=100 * sub.contribution,
            y=[labels.get(k, k) for k in sub.lever],
            orientation="h",
            marker={"color": [C_SCEN if v > 0 else C_REF for v in sub.contribution]},
            hovertemplate="%{y}: %{x:.2f} pp<extra></extra>",
        )
    )
    fig.update_layout(height=max(220, 26 * len(sub) + 80))
    _layout(fig, title, "")
    fig.update_xaxes(title_text="pp", gridcolor=GRID, showgrid=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(hovermode="closest")
    return fig


def status_rows(status: pd.DataFrame, targets: list[TargetSpec], lang: str) -> list[dict]:
    """Rows of the target table: label, year, target, model, margins, met / missed (icon + text)."""
    rows = []
    for (_, r), tg in zip(
        status.iterrows(),
        [x for x in targets if (x.quantity, x.year) in set(zip(status.quantity, status.year, strict=True))],
        strict=True,
    ):
        pct = tg.relative_to is not None
        fmt = (lambda v: f"{100 * v:+.1f} %") if pct else (lambda v: f"{v:,.3g}")
        mfmt = (lambda v: f"{100 * v:+.1f} pp") if pct else (lambda v: f"{v:+,.3g}")
        met = r.margin >= -1e-6
        row = {
            "label": tr(tg.label, lang),
            "year": tg.year,
            "kind": t("hard" if tg.hard else "reported", lang),
            "target": fmt(tg.value),
            "model": fmt(r.model),
            "margin": mfmt(r.margin),
            "status": ("✓ " + t("met", lang)) if met else ("✗ " + t("missed", lang)),
            "met": bool(met),
        }
        for c in status.columns:
            if c.startswith("margin_") and not pd.isna(r[c]):
                row[c] = mfmt(r[c])
        rows.append(row)
    return rows


# ---- app ---------------------------------------------------------------------------------------------------


def create_strategy_app(adapter: StrategyAdapter, lang: str = "de", title: str | None = None):
    """Dash app around a city adapter; the page language can be switched (de / en)."""
    from dash import ALL, Dash, Input, Output, State, dcc, html, no_update

    app = Dash(__name__, title=title or tr(adapter.title, lang), suppress_callback_exceptions=True)
    levers = {lv.key: lv for lv in adapter.levers}

    @lru_cache(maxsize=256)
    def _run(mkey: tuple, okey: tuple) -> pd.DataFrame:
        return adapter.run(dict(mkey), dict(okey))

    @lru_cache(maxsize=64)
    def _climate(mkey: tuple, okey: tuple):
        return adapter.run_climate_band(dict(mkey), dict(okey))

    def keys(mults: dict, opts: dict) -> tuple[tuple, tuple]:
        return tuple(sorted((k, round(float(v), 4)) for k, v in mults.items())), tuple(sorted(opts.items()))

    def chip(text: str, colour: str) -> html.Span:
        return html.Span(
            text,
            style={
                "border": f"1px solid {colour}",
                "color": INK,
                "borderRadius": "8px",
                "padding": "0 6px",
                "fontSize": "11px",
                "marginRight": "4px",
                "borderLeft": f"6px solid {colour}",
            },
        )

    def lever_panel(mults: dict, lg: str):
        out = [html.H4(t("levers", lg), style={"margin": "4px 0 8px 0"})]
        for sector, slabel in adapter.sectors.items():
            items = []
            for lv in [x for x in adapter.levers if x.sector == sector]:
                m = float(mults.get(lv.key, 1.0))
                marks = {lv.lo: f"{lv.lo:g}", 1: "1", lv.hi: f"{lv.hi:g}"} if lv.lo < 1 < lv.hi else None
                items.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.B(tr(lv.label, lg)),
                                    " ",
                                    chip(t("effect_" + lv.effect, lg), EFFECT_COLOUR.get(lv.effect, C_REF)),
                                    chip(t("conf_" + lv.confidence, lg), "#c2c2c2"),
                                    html.Span(
                                        (t("gaps", lg) + ": " + ", ".join(lv.gaps)) if lv.gaps else "",
                                        style={"fontSize": "11px", "color": INK2},
                                    ),
                                ]
                            ),
                            dcc.Slider(
                                id={"type": "lever", "key": lv.key},
                                min=lv.lo,
                                max=lv.hi,
                                step=lv.step,
                                value=m,
                                marks=marks,
                                updatemode="mouseup",
                                tooltip={"placement": "bottom"},
                            ),
                            html.Div(
                                adapter.level_text(lv.key, m, lg),
                                id={"type": "cap", "key": lv.key},
                                style={"fontSize": "11px", "color": INK2, "marginBottom": "8px"},
                            ),
                        ]
                    )
                )
            out.append(
                html.Details(
                    [html.Summary(tr(slabel, lg), style={"fontWeight": 600, "cursor": "pointer"}), *items],
                    open=True,
                    style={"marginBottom": "6px"},
                )
            )
        return out

    def gaps_table(lg: str):
        g = adapter.data_gaps
        if g is None or g.empty:
            return html.Div()
        used = {x for lv in adapter.levers for x in lv.gaps}
        g = g[g.id.isin(used)] if used else g
        head = ["id", "item", "current_value", "importance", "informant"]
        return html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in head]))]
            + [html.Tr([html.Td(str(r[h])) for h in head]) for _, r in g.iterrows()],
            style={"fontSize": "12px", "borderCollapse": "collapse", "width": "100%"},
        )

    def page(lg: str, mults: dict, opts: dict):
        option_controls = [
            html.Div(
                [
                    html.Label(tr(o.label, lg), style={"fontSize": "12px"}),
                    dcc.Dropdown(
                        id={"type": "opt", "key": o.key},
                        options=[{"label": tr(lab, lg), "value": v} for v, lab in o.choices],
                        value=opts.get(o.key, o.default),
                        clearable=False,
                    ),
                ],
                style={"minWidth": "200px", "marginRight": "12px"},
            )
            for o in adapter.options
        ]
        band_opts = []
        if adapter.response_bands:
            band_opts.append({"label": t("band_response", lg), "value": "response"})
        if adapter.climate_series:
            band_opts.append({"label": t("band_climate", lg), "value": "climate"})
        header = html.Div(
            [
                html.H2(tr(adapter.title, lg), style={"margin": "0 16px 0 0"}),
                html.Div(
                    [
                        html.Label(t("preset", lg), style={"fontSize": "12px"}),
                        dcc.Dropdown(
                            id="preset",
                            options=[
                                {"label": tr(p["label"], lg), "value": k} for k, p in adapter.presets.items()
                            ],
                            value=None,
                        ),
                    ],
                    style={"minWidth": "260px", "marginRight": "12px"},
                ),
                *option_controls,
                html.Div(
                    [
                        html.Label(t("bands", lg), style={"fontSize": "12px"}),
                        dcc.Checklist(id="bands", options=band_opts, value=[], style={"fontSize": "12px"}),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "alignItems": "flex-end",
                "gap": "4px",
                "marginBottom": "8px",
            },
        )
        tabs = dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(
                    label=t("tab_overview", lg),
                    value="overview",
                    children=[
                        html.Div(id="status-line", style={"margin": "8px 0"}),
                        dcc.Loading(
                            html.Div(
                                id="overview-graphs",
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                                },
                            )
                        ),
                    ],
                ),
                dcc.Tab(
                    label=t("tab_targets", lg),
                    value="targets",
                    children=[dcc.Loading(html.Div(id="target-table"))],
                ),
                dcc.Tab(
                    label=t("tab_bands", lg),
                    value="bands",
                    children=[dcc.Loading(html.Div(id="band-graphs"))],
                ),
                dcc.Tab(
                    label=t("tab_contrib", lg),
                    value="contrib",
                    children=[
                        html.P(t("contrib_note", lg), style={"fontSize": "12px", "color": INK2}),
                        html.Button(t("compute", lg), id="contrib-run"),
                        dcc.Loading(html.Div(id="contrib-graphs")),
                    ],
                ),
                dcc.Tab(
                    label=t("tab_compare", lg),
                    value="compare",
                    children=[
                        html.Label(t("compare_pick", lg), style={"fontSize": "12px"}),
                        dcc.Dropdown(
                            id="compare-pick",
                            multi=True,
                            value=[],
                            options=[
                                {"label": tr(p["label"], lg), "value": k} for k, p in adapter.presets.items()
                            ],
                        ),
                        dcc.Loading(html.Div(id="compare-out")),
                    ],
                ),
                dcc.Tab(label=t("tab_gaps", lg), value="gaps", children=[gaps_table(lg)]),
            ],
        )
        return html.Div(
            [
                header,
                html.Div(
                    [
                        html.Div(
                            lever_panel(mults, lg),
                            style={
                                "flex": "0 0 360px",
                                "maxHeight": "88vh",
                                "overflowY": "auto",
                                "paddingRight": "12px",
                                "borderRight": f"1px solid {GRID}",
                            },
                        ),
                        html.Div(tabs, style={"flex": "1 1 auto", "paddingLeft": "12px", "minWidth": 0}),
                    ],
                    style={"display": "flex"},
                ),
            ]
        )

    default_opts = {o.key: o.default for o in adapter.options}
    app.layout = html.Div(
        [
            dcc.Store(id="lang", data=lang),
            dcc.Store(id="mults", data=adapter.defaults()),
            dcc.Store(id="opts", data=default_opts),
            html.Div(
                dcc.RadioItems(
                    id="lang-pick",
                    options=[{"label": "Deutsch", "value": "de"}, {"label": "English", "value": "en"}],
                    value=lang,
                    inline=True,
                ),
                style={"float": "right", "fontSize": "12px"},
            ),
            html.Div(id="page", children=page(lang, adapter.defaults(), default_opts)),
        ],
        style={"fontFamily": FONT, "color": INK, "padding": "12px", "background": "#ffffff"},
    )

    @app.callback(
        Output("page", "children"),
        Output("lang", "data"),
        Input("lang-pick", "value"),
        State("mults", "data"),
        State("opts", "data"),
        prevent_initial_call=True,
    )
    def _language(lg, mults, opts):
        return page(lg, mults, opts), lg

    @app.callback(
        Output({"type": "lever", "key": ALL}, "value"),
        Output({"type": "opt", "key": ALL}, "value"),
        Input("preset", "value"),
        State({"type": "lever", "key": ALL}, "id"),
        State({"type": "opt", "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _preset(name, lever_ids, opt_ids):
        if not name:
            return no_update, no_update
        p = adapter.presets[name]
        mm = {**adapter.defaults(), **p.get("multipliers", {})}
        oo = {**default_opts, **p.get("options", {})}
        return [mm[i["key"]] for i in lever_ids], [oo[i["key"]] for i in opt_ids]

    @app.callback(
        Output("mults", "data"),
        Output("opts", "data"),
        Output({"type": "cap", "key": ALL}, "children"),
        Input({"type": "lever", "key": ALL}, "value"),
        Input({"type": "opt", "key": ALL}, "value"),
        State({"type": "lever", "key": ALL}, "id"),
        State({"type": "opt", "key": ALL}, "id"),
        State("lang", "data"),
    )
    def _collect(values, ovalues, lever_ids, opt_ids, lg):
        mults = {i["key"]: float(v) for i, v in zip(lever_ids, values, strict=True)}
        opts = {i["key"]: v for i, v in zip(opt_ids, ovalues, strict=True)}
        caps = [adapter.level_text(i["key"], mults[i["key"]], lg) for i in lever_ids]
        return mults, opts, caps

    def results(mults, opts, bands):
        mk, ok = keys(mults, opts)
        central = _run(mk, ok)
        band_runs = {}
        if "response" in (bands or []):
            for b in adapter.response_bands:
                band_runs[b] = _run(mk, tuple(sorted({**opts, "band": b}.items())))
        climate = _climate(mk, ok) if "climate" in (bands or []) else None
        return central, band_runs, climate

    @app.callback(
        Output("status-line", "children"),
        Output("overview-graphs", "children"),
        Output("target-table", "children"),
        Output("band-graphs", "children"),
        Input("mults", "data"),
        Input("opts", "data"),
        Input("bands", "value"),
        State("lang", "data"),
    )
    def _update(mults, opts, bands, lg):
        central, band_runs, climate = results(mults, opts, bands)
        status = target_status(central, adapter.targets, band_runs, climate)
        rows = status_rows(status, adapter.targets, lg)
        hard = [r for r in rows if r["kind"] == t("hard", lg)]
        tiles = [
            html.Span(
                f"{r['status']}: {r['label']} {r['year']} ({r['margin']})",
                style={
                    "border": f"1px solid {GOOD if r['met'] else CRITICAL}",
                    "borderRadius": "8px",
                    "padding": "2px 8px",
                    "marginRight": "6px",
                    "fontSize": "12px",
                    "display": "inline-block",
                    "marginBottom": "4px",
                },
            )
            for r in hard
        ]
        graphs = [
            dcc.Graph(
                figure=fig_series(s, central, lg, band_runs, adapter.references, adapter.targets),
                config={"displaylogo": False},
            )
            for s in adapter.pathways
        ]
        graphs += [
            dcc.Graph(
                figure=fig_stack(
                    parts,
                    central,
                    tr(adapter.sectors.get(k, k), lg) if k in adapter.sectors else k,
                    lg,
                    tr(parts[0].unit, lg),
                ),
                config={"displaylogo": False},
            )
            for k, parts in adapter.stacks.items()
        ]
        cols = (
            ["label", "year", "kind", "target", "model", "margin"]
            + sorted({c for r in rows for c in r if c.startswith("margin_")})
            + ["status"]
        )
        names = {
            "label": t("target", lg),
            "year": t("year", lg),
            "kind": "",
            "target": t("target", lg),
            "model": t("model", lg) + " (" + t("central", lg) + ")",
            "margin": t("margin", lg),
            "status": "",
            "margin_climate_min": t("margin", lg) + ": " + t("climate_min", lg),
        }
        for b in adapter.response_bands:
            names[f"margin_{b}"] = f"{t('margin', lg)}: {b}"
        table = html.Table(
            [html.Thead(html.Tr([html.Th(names.get(c, c)) for c in cols]))]
            + [
                html.Tr(
                    [
                        html.Td(
                            r.get(c, ""),
                            style={"color": (GOOD if r["met"] else CRITICAL) if c == "status" else INK},
                        )
                        for c in cols
                    ]
                )
                for r in rows
            ],
            style={"fontSize": "12px", "borderCollapse": "collapse", "width": "100%"},
        )
        band_graphs = [
            dcc.Graph(figure=fig_series(s, central, lg, band_runs, None, adapter.targets))
            for s in adapter.pathways
        ]
        if climate is not None and not climate.empty:
            band_graphs += [
                dcc.Graph(figure=fig_climate(s, climate, central, lg)) for s in adapter.climate_series
            ]
        elif "climate" in (bands or []):
            band_graphs.append(html.P(t("no_climate", lg)))
        return tiles, graphs, table, band_graphs

    @app.callback(
        Output("contrib-graphs", "children"),
        Input("contrib-run", "n_clicks"),
        State("mults", "data"),
        State("opts", "data"),
        State("lang", "data"),
        prevent_initial_call=True,
    )
    def _contrib(_, mults, opts, lg):
        hard = [tg for tg in adapter.targets if tg.hard]
        df = contributions(adapter, mults, opts, hard)
        labels = {k: tr(lv.label, lg) for k, lv in levers.items()}
        return [
            dcc.Graph(
                figure=fig_contributions(df, tg.quantity, tg.year, labels, f"{tr(tg.label, lg)} {tg.year}")
            )
            for tg in hard
            if not df[(df.quantity == tg.quantity) & (df.year == tg.year)].empty
        ]

    @app.callback(
        Output("compare-out", "children"),
        Input("compare-pick", "value"),
        State("mults", "data"),
        State("opts", "data"),
        State("lang", "data"),
    )
    def _compare(picked, mults, opts, lg):
        scen = {t("current", lg): (mults, opts)}
        for k in (picked or [])[:3]:
            p = adapter.presets[k]
            scen[tr(p["label"], lg)] = (
                {**adapter.defaults(), **p.get("multipliers", {})},
                {**default_opts, **p.get("options", {})},
            )
        res = {name: _run(*keys(m, o)) for name, (m, o) in scen.items()}
        figs = []
        for s in adapter.pathways:
            fig = go.Figure()
            for i, (name, r) in enumerate(res.items()):
                y = display(r, s)
                fig.add_trace(
                    go.Scatter(
                        x=list(y.index),
                        y=y.values,
                        name=name,
                        line={"color": SERIES[i % len(SERIES)], "width": 2},
                    )
                )
            figs.append(dcc.Graph(figure=_layout(fig, tr(s.label, lg), tr(s.unit, lg))))
        head = [t("levers", lg)] + list(scen)
        body = [
            html.Tr(
                [html.Td(tr(lv.label, lg))]
                + [html.Td(f"{float(m.get(lv.key, 1.0)):.2f}") for m, _ in scen.values()]
            )
            for lv in adapter.levers
        ]
        table = html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in head]))] + body,
            style={"fontSize": "12px", "borderCollapse": "collapse"},
        )
        return [
            html.Div(
                figs, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))"}
            ),
            table,
        ]

    return app


__all__ = [
    "create_strategy_app",
    "fig_series",
    "fig_stack",
    "fig_climate",
    "fig_contributions",
    "status_rows",
    "MILESTONES",
]
