"""Dash explorer: callbacks through the Dash HTTP interface (synthetic model, box geometry)."""

import json
import warnings
from functools import partial

import numpy as np
import pytest
import shapely

pytest.importorskip("dash")

from regsim.api.runtime import Fitted, short_id  # noqa: E402
from regsim.app.dash_app import create_dash  # noqa: E402
from regsim.models.base import ModelData  # noqa: E402
from regsim.models.hier_mf import HierTypedMF  # noqa: E402
from regsim.synthetic import generate_hierarchical  # noqa: E402
from regsim.uncertainty.conformal import Conformalised  # noqa: E402

warnings.filterwarnings("ignore")
WINDOWS = ("2013/2017", "2018/2022")


class _Profiles:
    """Stand-in panel model: its completed profiles are the panel values."""

    def __init__(self, Z):
        self.Z = Z

    def predict(self):
        return self.Z


def _panel(d: ModelData) -> Fitted:
    """Geo-time entities of ``d`` in two windows, the second shifted, as a panel model for the trajectories."""
    rng = np.random.default_rng(0)
    Z = np.nan_to_num(d.Z)
    Z2 = np.vstack([Z, Z + 0.3 + rng.normal(0, 0.1, Z.shape)])
    n = len(d.geo_ids)
    data = ModelData(
        Z=Z2,
        mask=np.ones_like(Z2, dtype=bool),
        parent=[[] for _ in range(2 * n)],
        agg=None,
        column_meta=d.column_meta,
        level=np.concatenate([d.levels(), d.levels()]),
        geo_ids=[f"{g}@{w}" for w in WINDOWS for g in d.geo_ids],
        column_ids=d.column_ids,
        period=np.repeat(WINDOWS, n),
    )
    return Fitted(_Profiles(Z2), data, None)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    d = generate_hierarchical(
        n_per_level=(4, 20, 80),
        n_indicators=8,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        seed=4,
    ).model_data()
    model = Conformalised(partial(HierTypedMF, rank=3, lam_u=0.3)).fit(d)
    geometry = {
        short_id(g): shapely.box(3e6 + i * 2e4, 2e6, 3e6 + (i + 1) * 2e4, 2.1e6)
        for i, g in enumerate(d.geo_ids)
    }
    fitted = Fitted(model, d, tmp_path_factory.mktemp("fit"))
    names = {g: f"Region {i}" for i, g in enumerate(d.geo_ids)}
    labels = {c: f"Indicator {c}" for c in d.column_ids}
    sources = {c: ("d36" if i % 2 else "eurostat") for i, c in enumerate(d.column_ids)}
    app = create_dash(
        fitted,
        geometry,
        panel=lambda: _panel(d),
        region_names=names,
        indicator_labels=labels,
        indicator_sources=sources,
        n_arrows=4,
    )
    return app, d


def _call(client, outputs, inputs, state=(), changed=None):
    payload = {
        "output": ".." + "...".join(f"{i}.{p}" for i, p in outputs) + ".."
        if len(outputs) > 1
        else f"{outputs[0][0]}.{outputs[0][1]}",
        "outputs": [{"id": i, "property": p} for i, p in outputs]
        if len(outputs) > 1
        else {"id": outputs[0][0], "property": outputs[0][1]},
        "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
        "state": [{"id": i, "property": p, "value": v} for i, p, v in state],
        "changedPropIds": [changed or f"{inputs[0][0]}.{inputs[0][1]}"],
    }
    r = client.post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200, r.data[:500]
    return r.get_json()["response"]


def _select(
    client, state, sel_map=None, sel_emb=None, click_map=None, click_emb=None, current=(), changed=""
):
    """Selection callback; None when it leaves the selection unchanged."""
    payload = {
        "output": "selection.data",
        "outputs": {"id": "selection", "property": "data"},
        "inputs": [
            {"id": "map", "property": "selectedData", "value": sel_map},
            {"id": "embedding", "property": "selectedData", "value": sel_emb},
            {"id": "map", "property": "clickData", "value": click_map},
            {"id": "embedding", "property": "clickData", "value": click_emb},
            {"id": "clear", "property": "n_clicks", "value": 1},
            {"id": "state", "property": "data", "value": state},
        ],
        "state": [{"id": "selection", "property": "data", "value": list(current)}],
        "changedPropIds": [changed],
    }
    r = client.post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")
    if r.status_code == 204:  # no_update
        return None
    assert r.status_code == 200, r.data[:500]
    return r.get_json()["response"].get("selection", {}).get("data")


def _compute(client, level, cols=(), method="pca", arrows=()):
    return _call(
        client,
        [("state", "data"), ("summary", "children"), ("reference", "options"), ("partner", "options")],
        [
            ("level", "value", level),
            ("lens", "value", list(cols)),
            ("k", "value", 3),
            ("method", "value", method),
            ("arrows", "value", list(arrows)),
        ],
    )


def _figures(client, state, ref=None, selection=(), partner=None):
    return _call(
        client,
        [("map", "figure"), ("embedding", "figure")],
        [
            ("state", "data", state),
            ("colour", "value", "type"),
            ("reference", "value", ref),
            ("partner", "value", partner),
            ("selection", "data", list(selection)),
        ],
    )


def test_layout_names_and_linked_selection(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    assert client.get("/").status_code == 200
    layout = repr(dash_app.layout)
    for cid in (
        "map",
        "embedding",
        "neighbours",
        "difference",
        "lens",
        "level",
        "arrows",
        "trajectories",
        "reference",
        "selection-info",
        "clear",
    ):
        assert f"id='{cid}'" in layout
    assert "[Eurostat] Indicator ind_0 (ind_0)" in layout  # readable lens options with source
    assert "KNOWING D3.6 only (4)" in layout
    out = _compute(client, 2)
    state = out["state"]["data"]
    assert (
        len(state["ids"]) == int((d.levels() == 2).sum()) and "trustworthiness" in out["summary"]["children"]
    )
    assert state["directions"] == []
    options = out["reference"]["options"]
    assert len(options) == len(state["ids"]) and options[0]["label"].startswith("Region ")
    ids = state["ids"][:3]
    select = partial(_select, client, state)
    sel = select(sel_emb={"points": [{"customdata": g} for g in ids]}, changed="embedding.selectedData")
    assert sel == sorted(ids)
    info = _call(
        client,
        [("selection-info", "children")],
        [("selection", "data", sel), ("reference", "value", ids[0]), ("partner", "value", ids[1])],
    )
    text = json.dumps(info["selection-info"]["children"])
    assert (
        "Reference" in text
        and "compared with" in text
        and "Selected regions: " in text
        and "3: Region " in text
    )
    # empty events (figure redrawn, click in lasso mode) keep the selection; a click adds; clear empties
    assert select(sel_emb=None, current=sel, changed="embedding.selectedData") is None
    assert select(sel_map={"points": []}, current=sel, changed="map.selectedData") is None
    other = state["ids"][5]
    added = select(click_map={"points": [{"customdata": other}]}, current=sel, changed="map.clickData")
    assert added == sorted([*ids, other])
    assert select(current=sel, changed="clear.n_clicks") == []
    assert select(current=[*sel, "elsewhere"], changed="state.data") == sel
    figs = _figures(client, state, ref=ids[0], selection=sel)
    centroids = [t for t in figs["map"]["figure"]["data"] if t.get("customdata")]
    assert sorted(centroids[0]["selectedpoints"]) == sorted(centroids[0]["customdata"].index(g) for g in ids)
    outline = [t for t in figs["map"]["figure"]["data"] if t.get("line", {}).get("color") == "red"]
    assert len(outline) == 1 and outline[0]["x"].count(None) == 3  # one ring per selected box
    assert (
        figs["map"]["figure"]["layout"]["uirevision"] == figs["embedding"]["figure"]["layout"]["uirevision"]
    )
    assert any(
        "Region " in str(t.get("text")) for t in figs["embedding"]["figure"]["data"]
    )  # reference label


def test_neighbours_and_difference(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    state = _compute(client, 2)["state"]["data"]
    ref = state["ids"][0]
    nb = _call(
        client,
        [("neighbours", "data"), ("neighbours", "columns")],
        [("reference", "value", ref), ("state", "data", state)],
        state=[("lens", "value", [])],
    )
    rows = nb["neighbours"]["data"]
    assert len(rows) == 10 and rows[0]["region"].startswith("Region ")
    assert "id" not in [c["id"] for c in nb["neighbours"]["columns"]]
    diff = _call(
        client,
        [("difference", "figure")],
        [("reference", "value", ref), ("partner", "value", rows[2]["id"])],
        state=[("lens", "value", [])],
    )["difference"]["figure"]
    title = diff["layout"]["title"]["text"]
    assert rows[2]["region"] in title and "rank 3 of " in title and diff["data"]
    far = state["ids"][-1]  # any region of the level, not only the nearest ones
    diff = _call(
        client,
        [("difference", "figure")],
        [("reference", "value", ref), ("partner", "value", far)],
        state=[("lens", "value", [])],
    )["difference"]["figure"]
    assert "Region " in diff["layout"]["title"]["text"] and "rank " in diff["layout"]["title"]["text"]
    marked = _call(
        client,
        [("neighbours", "style_data_conditional")],
        [("partner", "value", rows[2]["id"]), ("neighbours", "data", rows)],
    )["neighbours"]["style_data_conditional"]
    assert [m["if"]["row_index"] for m in marked] == [2]


def _pick(client, changed, ref=None, partner=None, next_click="reference", map_click=None, cell=None):
    """Click/table/search callback; returns (reference, partner, next click), None where unchanged."""
    payload = {
        "output": "..reference.value...partner.value...click-next.data..",
        "outputs": [
            {"id": "reference", "property": "value"},
            {"id": "partner", "property": "value"},
            {"id": "click-next", "property": "data"},
        ],
        "inputs": [
            {"id": "map", "property": "clickData", "value": map_click},
            {"id": "embedding", "property": "clickData", "value": None},
            {"id": "neighbours", "property": "active_cell", "value": cell},
            {"id": "reference", "property": "value", "value": ref},
            {"id": "partner", "property": "value", "value": partner},
        ],
        "state": [
            {"id": "click-next", "property": "data", "value": next_click},
            {"id": "lens", "property": "value", "value": []},
        ],
        "changedPropIds": [changed],
    }
    r = client.post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")
    if r.status_code == 204:
        return None, None, None
    assert r.status_code == 200, r.data[:500]
    out = r.get_json()["response"]
    return (
        out.get("reference", {}).get("value"),
        out.get("partner", {}).get("value"),
        out.get("click-next", {}).get("data"),
    )


def test_clicks_alternate_between_reference_and_comparison(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    a, b, c = [g for g, v in zip(d.geo_ids, d.levels(), strict=True) if v == 2][:3]

    def click(g):
        return {"points": [{"customdata": g}]}

    ref, nearest_a, after = _pick(client, "map.clickData", map_click=click(a))
    assert (ref, after) == (a, "partner") and nearest_a in d.geo_ids and nearest_a != a  # nearest by default
    assert _pick(client, "map.clickData", ref=a, next_click="partner", map_click=click(b)) == (
        None,
        b,
        "reference",
    )
    assert _pick(client, "map.clickData", ref=a, partner=b, map_click=click(c))[::2] == (c, "partner")
    cell = {"row": 1, "column": 0, "column_id": "region", "row_id": c}
    assert _pick(client, "neighbours.active_cell", ref=a, cell=cell) == (None, c, "reference")
    assert _pick(client, "reference.value", ref=a) == (None, nearest_a, "partner")  # typed reference
    hint = _call(
        client, [("click-hint", "children")], [("click-next", "data", "partner"), ("reference", "value", a)]
    )
    assert "comparison region" in json.dumps(hint)


def test_shared_coordinate_system_and_directions(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    out = _compute(client, 2, arrows=["show"])
    state = out["state"]["data"]
    assert "both period windows" in out["summary"]["children"]
    assert len(state["directions"]) == 4  # n_arrows
    lens = d.column_ids[:3]
    own = _compute(client, 2, cols=lens, arrows=["show", "lens"])["state"]["data"]["directions"]
    assert [a["name"] for a in own] == lens
    figs = _figures(client, state)
    labels = [a["text"] for a in figs["embedding"]["figure"]["layout"]["annotations"] if a.get("text")]
    assert len(labels) == 4  # one arrow and one label per indicator
    traj = _call(
        client,
        [("trajectories", "figure")],
        [
            ("tabs", "value", "traj"),
            ("traj-opts", "value", ["arrows"]),
            ("selection", "data", []),
            ("level", "value", 2),
            ("lens", "value", []),
        ],
        state=[("arrows", "value", ["show"])],
    )["trajectories"]["figure"]
    heads = {
        a["hovertext"].split("(")[1].split(")")[0]: (round(a["x"], 6), round(a["y"], 6))
        for a in traj["layout"]["annotations"]
        if a.get("hovertext") and "Indicator" in a["hovertext"]
    }
    assert heads == {a["name"]: (round(a["x"], 6), round(a["y"], 6)) for a in state["directions"]}


def test_trajectories_tab(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    ids = [g for g, v in zip(d.geo_ids, d.levels(), strict=True) if v == 2][:5]

    def traj(opts, selection, lens=()):
        return _call(
            client,
            [("trajectories", "figure")],
            [
                ("tabs", "value", "traj"),
                ("traj-opts", "value", opts),
                ("selection", "data", selection),
                ("level", "value", 2),
                ("lens", "value", list(lens)),
            ],
            state=[("arrows", "value", [])],
        )["trajectories"]["figure"]

    followed = traj(["follow"], ids)
    texts = [a["text"] for a in followed["layout"]["annotations"] if a.get("text")]
    assert len(texts) == 5 and all(x.startswith("Region ") for x in texts)
    assert "5 selected regions" in followed["layout"]["title"]["text"]
    assert "select regions" in traj(["follow"], [])["layout"]["title"]["text"]
    assert "relative to mean drift" in traj(["follow", "relative"], ids)["layout"]["title"]["text"]
    assert "capital regions" in traj([], ids)["layout"]["title"]["text"]
    assert traj(["follow"], ids, lens=d.column_ids[:3])["data"]  # lens-specific basis


def test_sources_and_short_labels(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    d36 = _call(
        client,
        [("lens", "value")],
        [("src-all", "n_clicks", 0), ("src-eurostat", "n_clicks", 0), ("src-d36", "n_clicks", 1)],
        changed="src-d36.n_clicks",
    )["lens"]["value"]
    assert d36 == [c for i, c in enumerate(d.column_ids) if i % 2]
    everything = _call(
        client,
        [("lens", "value")],
        [("src-all", "n_clicks", 1), ("src-eurostat", "n_clicks", 0), ("src-d36", "n_clicks", 1)],
        changed="src-all.n_clicks",
    )["lens"]["value"]
    assert everything == []
    out = _compute(client, 2, cols=d36, arrows=["show", "lens"])
    assert "KNOWING D3.6 4" in out["summary"]["children"]
    figs = _figures(client, out["state"]["data"])
    notes = [a for a in figs["embedding"]["figure"]["layout"]["annotations"] if a.get("text")]
    assert all(a["text"].startswith("Indicator ") and "KNOWING D3.6" in a["hovertext"] for a in notes)
    state = _compute(client, 2)["state"]["data"]
    rows = _call(
        client,
        [("neighbours", "data"), ("neighbours", "columns")],
        [("reference", "value", state["ids"][0]), ("state", "data", state)],
        state=[("lens", "value", [])],
    )["neighbours"]["data"]
    assert "Indicator ind_" in rows[0]["differs most in"]
    diff = _call(
        client,
        [("difference", "figure")],
        [("reference", "value", state["ids"][0]), ("partner", "value", rows[0]["id"])],
        state=[("lens", "value", [])],
    )["difference"]["figure"]
    assert all(y.startswith("Indicator ") for y in diff["data"][0]["y"])
