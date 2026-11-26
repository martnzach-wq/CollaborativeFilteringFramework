"""Curated tables, composition transfer and coping-type estimates (decision D54) on synthetic data."""

import numpy as np
import pandas as pd

from regsim.causal import coping as C
from regsim.data.table import TableConnector
from regsim.similarity.transfer import estimate_composition


def _table(tmp_path):
    rows = []
    shares = {
        "AA": [0.10, 0.10, 0.30, 0.20, 0.20, 0.10],
        "BB": [0.02, 0.08, 0.10, 0.40, 0.30, 0.10],
        "AA001": [0.05, 0.10, 0.30, 0.20, 0.25, 0.10],
    }
    for geo, v in shares.items():
        flag = "national sample" if len(geo) == 2 else "metropolitan column of the national sample"
        for t, x in zip(C.TYPES, v, strict=True):
            rows.append([geo, f"coping_share_{t}", "2023", x, "knowing", "test", flag])
    path = tmp_path / "coping.csv"
    pd.DataFrame(
        rows, columns=["geo_code", "indicator_id", "period", "value", "source", "source_version", "flags"]
    ).to_csv(path, index=False)
    return path


def test_table_connector_returns_one_indicator(tmp_path):
    path = _table(tmp_path)
    obs = TableConnector(root=tmp_path).fetch({"dataset": path.name, "indicator_id": "coping_share_d"})
    assert set(obs.geo_code) == {"AA", "BB", "AA001"}
    assert (obs.scenario == "observed").all()
    assert obs.loc[obs.geo_code == "BB", "value"].item() == 0.40


def test_observed_shares_and_settlement_offset(tmp_path):
    shares, kind = C.observed(_table(tmp_path))
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert kind["AA001"] == "metropolitan" and kind["AA"] == "national"
    off = C.settlement_offsets(shares, kind)["metropolitan"]
    assert abs(off.sum()) < 1e-9  # clr differences sum to zero


def test_estimate_composition_weights_and_closure():
    rng = np.random.default_rng(0)
    donors = [f"D{i}" for i in range(6)]
    raw = rng.dirichlet(np.ones(4) * 3, size=6)
    shares = pd.DataFrame(raw, index=donors, columns=list("wxyz"))
    feats = pd.DataFrame(rng.normal(size=(7, 2)), index=donors + ["T"], columns=["f1", "f2"])
    r = estimate_composition(shares, feats, ["T"])
    assert np.isclose(r.estimate.loc["T"].sum(), 1.0)
    assert np.isclose(r.weights.loc["T"].sum(), 1.0)
    assert (r.lower.loc["T"] <= r.estimate.loc["T"] + 1e-12).all()
    assert (r.upper.loc["T"] >= r.estimate.loc["T"] - 1e-12).all()
    # a target at the features of one donor, with a narrow kernel, is close to that donor
    feats.loc["T"] = feats.loc["D2"]
    near = estimate_composition(shares, feats, ["T"], bandwidths=(0.05,))
    assert np.allclose(near.estimate.loc["T"], shares.loc["D2"], atol=1e-3)
