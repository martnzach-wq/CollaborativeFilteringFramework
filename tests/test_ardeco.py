from pathlib import Path

import pytest

from regsim.data.ardeco import ArdecoConnector, ArdecoError

FIX = Path(__file__).parent / "fixtures" / "ardeco"


class Response:
    def __init__(self, status_code, text, url="stub"):
        self.status_code, self.text, self.url = status_code, text, url


class StubSession:
    def __init__(self, fail_first=0):
        self.calls = []
        self.fail_first = fail_first

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", json))
        if "SUVGDP" in json["query"]:
            return Response(200, (FIX / "SUVGDP_meta.json").read_text(encoding="utf-8"))
        return Response(200, '{"data": {"variable": null}}')

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", dict(params)))
        if self.fail_first:
            self.fail_first -= 1
            return Response(500, "Internal Server Error")
        if dict(params)["level_id"] == "2":
            return Response(200, (FIX / "SUVGDP_L2.csv").read_text(encoding="utf-8"))
        return Response(200, "")


def conn(tmp_path, session):
    return ArdecoConnector(cache_dir=str(tmp_path), session=session, sleep=lambda s: None)


SPEC = {
    "dataset": "SUVGDP",
    "indicator_id": "ardeco_gdp",
    "filters": {"unit": "PPS_EU27_2020"},
    "since": 2018,
    "until": 2022,
}


def test_datasets_and_validation(tmp_path):
    c = conn(tmp_path, StubSession())
    assert {"unit": "PPS_EU27_2020"} in c.datasets("SUVGDP")
    c.validate("SUVGDP", {"unit": "PPS_EU27_2020"})
    with pytest.raises(ValueError, match="select 0"):
        c.validate("SUVGDP", {"unit": "XYZ"})
    with pytest.raises(ValueError, match="unknown dimensions"):
        c.validate("SUVGDP", {"sex": "TOTAL"})
    with pytest.raises(ArdecoError, match="unknown ARDECO variable"):
        c.datasets("NOPE")


def test_fetch_filters_years_and_flags(tmp_path):
    s = StubSession(fail_first=1)
    obs = conn(tmp_path, s).fetch({**SPEC, "levels": [2, 3]})
    assert set(obs.geo_code) == {"AT13", "PT1A", "DE30"}
    assert obs.period.astype(int).between(2018, 2022).all()
    assert len(obs) == 15
    assert (obs["flags"] == "ardeco").all() and (obs.source == "ardeco:SUVGDP").all()
    assert (obs.source_version == "NUTS2024").all()
    gets = [p for m, p in s.calls if m == "GET"]
    assert gets[0]["version"] == "2024" and gets[0]["unit"] == "PPS_EU27_2020"
    # Cached afterwards.
    n = len(s.calls)
    conn(tmp_path, s).fetch({**SPEC, "levels": [2, 3]})
    assert len(s.calls) == n
