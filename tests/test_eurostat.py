import json
from pathlib import Path

import numpy as np
import pytest

from regsim.data.eurostat import EurostatConnector, EurostatError, parse_jsonstat

FIX = Path(__file__).parent / "fixtures" / "eurostat"


class Response:
    def __init__(self, status_code, text, url="stub"):
        self.status_code, self.text, self.url = status_code, text, url


class StubSession:
    """Serves recorded responses; `queue` overrides the next responses for data requests."""

    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "contentconstraint" in url:
            return Response(200, (FIX / "demo_r_d3dens_constraint.xml").read_text(encoding="utf-8"))
        if self.queue:
            return self.queue.pop(0)
        return Response(200, (FIX / "demo_r_d3dens.json").read_text(encoding="utf-8"))


def connector(tmp_path, session, **kw):
    return EurostatConnector(cache_dir=str(tmp_path), session=session, sleep=lambda s: None, **kw)


SPEC = {"dataset": "demo_r_d3dens", "indicator_id": "pop_density", "filters": {"unit": "PER_KM2"}}


def test_structure_lists_dimensions(tmp_path):
    dims = connector(tmp_path, StubSession()).structure("demo_r_d3dens")
    assert {"freq", "unit", "geo", "time_period"} <= set(dims)
    assert "AT130" in dims["geo"] and dims["unit"] == ["PER_KM2"]


def test_validate_rejects_unknown_dimension_and_code(tmp_path):
    c = connector(tmp_path, StubSession())
    c.validate("demo_r_d3dens", {"unit": "PER_KM2", "geo": ["AT130"], "time": "2020"})
    with pytest.raises(ValueError, match="dimension 'currency'"):
        c.validate("demo_r_d3dens", {"currency": "EUR"})
    with pytest.raises(ValueError, match="XX"):
        c.validate("demo_r_d3dens", {"unit": ["XX"]})


def test_fetch_returns_long_observations(tmp_path):
    obs = connector(tmp_path, StubSession()).fetch(SPEC)
    assert len(obs) == 9
    assert set(obs.geo_code) == {"AT1", "AT13", "AT130"}
    row = obs[(obs.geo_code == "AT130") & (obs.period == "2021")].iloc[0]
    assert row.value == pytest.approx(4838.6)
    assert row.indicator_id == "pop_density" and row.source == "eurostat:demo_r_d3dens"
    assert row.source_version.startswith("20")
    assert "unit" not in obs.columns


def test_responses_are_cached(tmp_path):
    s = StubSession()
    connector(tmp_path, s).fetch(SPEC)
    n = len(s.calls)
    connector(tmp_path, s).fetch(SPEC)
    assert len(s.calls) == n
    offline = connector(tmp_path, StubSession(), offline=True)
    assert len(offline.fetch(SPEC)) == 9
    with pytest.raises(EurostatError, match="offline"):
        offline.fetch({**SPEC, "since": 2000, "validate": False})


def test_retries_413_and_asynchronous_responses(tmp_path):
    body = (FIX / "demo_r_d3dens.json").read_text(encoding="utf-8")
    asynchronous = json.dumps({"warning": {"status": 413, "label": "ASYNCHRONOUS_RESPONSE. Retry later."}})
    s = StubSession([Response(413, "too large"), Response(200, asynchronous), Response(200, body)])
    assert len(connector(tmp_path, s).fetch(SPEC)) == 9


def test_persistent_413_splits_by_period(tmp_path):
    body = (FIX / "demo_r_d3dens.json").read_text(encoding="utf-8")
    s = StubSession([Response(413, "too large")] * 3 + [Response(200, body)] * 2)
    obs = connector(tmp_path, s, max_retries=2).fetch({**SPEC, "since": 2019, "until": 2020})
    assert len(obs) == 18  # two yearly requests, each served the recorded response
    with pytest.raises(EurostatError, match="cannot be split"):
        s2 = StubSession([Response(413, "too large")] * 3)
        connector(tmp_path / "b", s2, max_retries=2).fetch(SPEC)


def test_http_error_is_raised(tmp_path):
    with pytest.raises(EurostatError, match="HTTP 404"):
        connector(tmp_path, StubSession([Response(404, "not found")])).fetch(SPEC)


def test_parse_jsonstat_flags_and_list_values():
    payload = {
        "id": ["geo", "time"],
        "size": [2, 2],
        "dimension": {
            "geo": {"category": {"index": {"A": 0, "B": 1}}},
            "time": {"category": {"index": ["2019", "2020"]}},
        },
        "value": [1.0, None, 3.0, 4.0],
        "status": {"2": "p", "3": "e"},
    }
    f = parse_jsonstat(payload)
    assert f[["geo", "time"]].values.tolist() == [["A", "2019"], ["B", "2019"], ["B", "2020"]]
    assert np.allclose(f.value, [1, 3, 4]) and f.flag.tolist() == ["", "p", "e"]
