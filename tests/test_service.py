"""Contract tests of the REST service (synthetic model; no store, no network)."""

import warnings
from functools import partial

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from regsim.api import service  # noqa: E402
from regsim.api.runtime import Fitted  # noqa: E402
from regsim.models.hier_mf import HierTypedMF  # noqa: E402
from regsim.synthetic import generate_hierarchical  # noqa: E402
from regsim.uncertainty.conformal import Conformalised  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = generate_hierarchical(
        n_per_level=(4, 20, 80),
        n_indicators=8,
        rank=3,
        noise=0.1,
        missingness="mcar",
        missing_rate=0.3,
        seed=3,
    ).model_data()
    model = Conformalised(partial(HierTypedMF, rank=3, lam_u=0.3)).fit(d)
    app = service.create_app(fitted=Fitted(model, d, tmp_path_factory.mktemp("fit")))
    return TestClient(app), d


def _check(resp, model, many=False):
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for item in body if many else [body]:
        model.model_validate(item)
    return body


def test_openapi_lists_all_endpoints(client):
    c, _ = client
    paths = set(c.get("/openapi.json").json()["paths"])
    assert paths >= {
        "/health",
        "/entities",
        "/indicators",
        "/predict/{geo_id}",
        "/similar/{geo_id}",
        "/entity",
        "/vision",
        "/next-indicator/{geo_id}",
        "/level-transfer/{indicator}",
        "/typology",
    }


def test_read_endpoints(client):
    c, d = client
    h = _check(c.get("/health"), service.Health)
    assert h["entities"] == len(d.geo_ids) and h["indicators"] == len(d.column_ids)
    ents = _check(c.get("/entities", params={"level": 2}), service.Entity, many=True)
    assert ents and {e["level"] for e in ents} == {2}
    assert {e["domain"] for e in ents} <= {"in_domain", "edge", "outside"}
    _check(c.get("/indicators"), service.IndicatorInfo, many=True)
    g = d.geo_ids[30]
    p = _check(c.get(f"/predict/{g}"), service.Prediction)
    assert len(p["values"]) == len(d.column_ids)
    one = _check(c.get(f"/predict/{g}", params={"indicator": d.column_ids[0]}), service.Prediction)
    assert len(one["values"]) == 1
    s = _check(c.get(f"/similar/{g}", params={"k": 4}), service.Similar)
    assert len(s["neighbours"]) == 4 and g not in {n["geo_id"] for n in s["neighbours"]}
    s2 = _check(
        c.get(f"/similar/{g}", params={"k": 3, "indicators": ",".join(d.column_ids[:3])}), service.Similar
    )
    assert s2["lens"] == d.column_ids[:3]


def test_task_endpoints(client):
    c, d = client
    g, c0, c1 = d.geo_ids[30], d.column_ids[0], d.column_ids[1]
    vals = _check(
        c.post("/entity", json={"values": {c0: 0.5}, "parent": d.geo_ids[5]}), service.Value, many=True
    )
    assert [v["observed"] for v in vals].count(True) == 1
    vis = _check(c.post("/vision", json={"base": g, "targets": {c0: 1.0}}), service.Vision)
    assert any(v["target"] for v in vis["values"]) and len(vis["analogues"]) == 5
    _check(c.get(f"/next-indicator/{g}", params={"targets": c1}), service.Candidate, many=True)
    lt = _check(c.get(f"/level-transfer/{c0}", params={"level": 2}), service.LevelValue, many=True)
    assert len(lt) == int((d.levels() == 2).sum())
    ty = _check(c.post("/typology", json={"level": 2, "k": 3, "n_boot": 2}), service.Typology)
    assert ty["k"] == 3 and len(ty["assignments"]) == int((d.levels() == 2).sum())


def test_errors(client):
    c, _ = client
    assert c.get("/predict/XX999").status_code == 404
    assert c.get("/predict/" + "SYN:L2_00", params={"indicator": "nope"}).status_code == 404
    assert c.get("/predict/SYN:L2_00", params={"alpha": 2}).status_code == 422
    assert c.post("/typology", json={"algorithm": "kmeans"}).status_code == 422
    assert c.post("/entity", json={"values": "x"}).status_code == 422
