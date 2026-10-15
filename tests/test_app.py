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
    return create_dash(fitted, geometry, panel=lambda: _panel(d), n_arrows=4), d


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


def _compute(client, level, cols=(), method="pca", arrows=()):
    return _call(
        client,
        [("state", "data"), ("summary", "children")],
        [
            ("level", "value", level),
            ("lens", "value", list(cols)),
            ("k", "value", 3),
            ("method", "value", method),
            ("arrows", "value", list(arrows)),
        ],
    )


def test_layout_and_linked_callbacks(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    assert client.get("/").status_code == 200
    layout = repr(dash_app.layout)
    for cid in ("map", "embedding", "neighbours", "difference", "lens", "level", "arrows", "trajectories"):
        assert f"id='{cid}'" in layout
    out = _compute(client, 2)
    state = out["state"]["data"]
    assert (
        len(state["ids"]) == int((d.levels() == 2).sum()) and "trustworthiness" in out["summary"]["children"]
    )
    assert state["directions"] == []
    ids = state["ids"][:3]
    sel = _call(
        client,
        [("selection", "data")],
        [
            ("map", "selectedData", None),
            ("embedding", "selectedData", {"points": [{"customdata": g} for g in ids]}),
        ],
        changed="embedding.selectedData",
    )["selection"]["data"]
    assert sel == sorted(ids)
    figs = _call(
        client,
        [("map", "figure"), ("embedding", "figure")],
        [
            ("state", "data", state),
            ("colour", "value", "type"),
            ("reference", "data", None),
            ("selection", "data", sel),
        ],
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


def test_indicator_directions(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    strongest = _compute(client, 2, arrows=["show"])
    assert len(strongest["state"]["data"]["directions"]) == 4  # n_arrows
    assert "biplot" in strongest["summary"]["children"]
    lens = d.column_ids[:3]
    own = _compute(client, 2, cols=lens, arrows=["show", "lens"])["state"]["data"]["directions"]
    assert [a["name"] for a in own] == lens
    figs = _call(
        client,
        [("map", "figure"), ("embedding", "figure")],
        [
            ("state", "data", strongest["state"]["data"]),
            ("colour", "value", "type"),
            ("reference", "data", None),
            ("selection", "data", []),
        ],
    )
    labels = [a["text"] for a in figs["embedding"]["figure"]["layout"]["annotations"] if a.get("text")]
    assert len(labels) == 4  # one arrow and one label per indicator


def test_trajectories_tab(app):
    dash_app, d = app
    client = dash_app.server.test_client()
    ids = [g for g, v in zip(d.geo_ids, d.levels(), strict=True) if v == 2][:5]

    def traj(opts, selection):
        return _call(
            client,
            [("trajectories", "figure")],
            [
                ("tabs", "value", "traj"),
                ("traj-opts", "value", opts),
                ("selection", "data", selection),
                ("level", "value", 2),
            ],
        )["trajectories"]["figure"]

    followed = traj(["follow"], ids)
    assert (
        len([a for a in followed["layout"]["annotations"] if a.get("text")]) == 5
        and "5 selected regions" in followed["layout"]["title"]["text"]
    )
    assert "select regions" in traj(["follow"], [])["layout"]["title"]["text"]
    relative = traj(["follow", "relative"], ids)
    assert "relative to mean drift" in relative["layout"]["title"]["text"]
    assert "capital regions" in traj([], ids)["layout"]["title"]["text"]
