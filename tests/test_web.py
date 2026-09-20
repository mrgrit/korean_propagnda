import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import opinion_field.web.app as web
    root = tmp_path_factory.mktemp("webroot")
    (root / "configs").mkdir()
    for f in ("campaign.yaml", "priors.yaml", "channels.yaml", "display_map.yaml"):
        (root / "configs" / f).write_text((ROOT / "configs" / f).read_text(encoding="utf-8"), encoding="utf-8")
    (root / "configs" / "web_auth.yaml").write_text("password: testpw\n", encoding="utf-8")
    (root / "runs").mkdir(); (root / "data").mkdir()
    web.ROOT = root
    return TestClient(web.app)


def test_auth_gate_and_login(client):
    assert client.get("/api/status").status_code == 401
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"
    r = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert r.status_code == 303 and "err=1" in r.headers["location"]
    r = client.post("/login", data={"password": "testpw"}, follow_redirects=False)
    assert r.status_code == 303 and "of_session" in r.cookies
    st = client.get("/api/status").json()
    assert "claude" in st and st["data"] == [] and st["busy"] is False


def test_config_roundtrip_and_validation(client):
    c = client.get("/api/config").json()
    assert c["config"]["rounds"] == 20
    r = client.put("/api/config", json={"overrides": {"rounds": 3, "defender.strength": 0.9}})
    assert r.status_code == 200 and r.json()["config"]["rounds"] == 3 and r.json()["config"]["defender"]["strength"] == 0.9
    bad = client.put("/api/files/campaign", json={"text": "rounds: 3\nnot_a_key: 1\n"})
    assert bad.status_code >= 400
    ok = client.put("/api/files/display_map", json={"text": "slots:\n  A: {display: 'X'}\n"})
    assert ok.status_code == 200


def test_run_validation(client):
    assert client.post("/api/runs", json={"name": "bad name!", "mode": "run"}).status_code == 400
    r = client.post("/api/runs", json={"name": "r1", "mode": "run", "data_dir": "data/nope"})
    assert r.status_code == 400
    assert client.get("/api/runs/none").status_code == 404
