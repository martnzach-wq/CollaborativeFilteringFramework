"""Dash explorer: callbacks through the Dash HTTP interface (synthetic model, box geometry)."""

import json
import warnings
from functools import partial

import pytest
import shapely

pytest.importorskip("dash")

from regsim.api.runtime import Fitted, short_id  # noqa: E402
from regsim.app.dash_app import create_dash  # noqa: E402
from regsim.models.hier_mf import HierTypedMF  # noqa: E402
from regsim.synthetic import generate_hierarchical  # noqa: E402
from regsim.uncertainty.conformal import Conformalised  # noqa: E402

warnings.filterwarnings("ignore")


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
    return create_dash(Fitted(model, d, tmp_path_factory.mktemp("fit")), geometry), d


def _call(client, outputs, inputs, state=(), changed=None):
    payload = {
        "output": ".." + "...".join(f"{i}.{p}" for i, p in outputs) + ".."
        if len(outputs) > 1
        else f"{outputs[0][0]}.{outputs[0][1]}",
        "outputs": [{"id": i, "property": p} for i, p in outputs],
        "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
        "state": [{"id": i, "property": p, "value": v} for i, p, v in state],
        "changedPropIds": [changed or f"{inputs[0][0]}.{inputs[0][1]}"],
    }
    r = client.post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200, r.data[:500]
    return r.get_json()["response"]


def test_layout_and_linked_callbacks(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    assert client.get("/").status_code == 200
    layout = repr(dash_app.layout)
    for cid in ("map", "embedding", "neighbours", "difference", "lens", "level"):
        assert f"id='{cid}'" in layout
    out = _call(
        client,
        [("state", "data"), ("summary", "children")],
        [("level", "value", 2), ("lens", "value", []), ("k", "value", 3), ("method", "value", "pca")],
    )
    state = out["state"]["data"]
    assert (
        len(state["ids"]) == int((d.levels() == 2).sum()) and "trustworthiness" in out["summary"]["children"]
    )
    ids = state["ids"][:3]
    sel = {"points": [{"customdata": g} for g in ids]}
    figs = _call(
        client,
        [("map", "figure"), ("embedding", "figure")],
        [
            ("state", "data", state),
            ("colour", "value", "type"),
            ("reference", "data", None),
            ("map", "selectedData", None),
            ("embedding", "selectedData", sel),
        ],
        changed="embedding.selectedData",
    )
    centroids = [t for t in figs["map"]["figure"]["data"] if t.get("customdata")]
    assert sorted(centroids[0]["selectedpoints"]) == sorted(centroids[0]["customdata"].index(g) for g in ids)
    det = _call(
        client,
        [("neighbours", "data"), ("neighbours", "columns"), ("difference", "figure")],
        [("reference", "data", ids[0]), ("state", "data", state)],
        state=[("lens", "value", [])],
    )
    assert len(det["neighbours"]["data"]) == 10 and det["difference"]["figure"]["data"]
