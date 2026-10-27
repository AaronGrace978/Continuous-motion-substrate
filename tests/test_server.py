from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from chronosoul_bridge import StaticAffectSource
from motion_substrate import ContinuousMotionSubstrate
from providers import ProviderSettings
from server import create_app


@pytest.fixture
def client(tmp_path):
    sub = ContinuousMotionSubstrate(str(tmp_path / "m.db"), affect_source=StaticAffectSource(0.1, 0.25),
                                    catch_up=False)
    app = create_app(sub, ProviderSettings(tmp_path / "p.json"), autostart=False)
    with TestClient(app) as c:
        yield c
    sub.store.close()


def test_index_serves_visualizer(client) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "<canvas" in r.text


def test_tick_and_state(client) -> None:
    r = client.post("/tick", json={"steps": 10, "target": [1.0, 0.0]})
    assert r.status_code == 200
    body = r.json()
    assert body["tick"] == 10 and body["position"][0] > 0.3
    state = client.get("/state").json()
    assert state["tick"] == 10
    assert set(state["params"]) >= {"stiffness", "mass", "damping_ratio", "momentum_gain", "jitter"}
    assert len(state["prediction"]["position"]) == 2


def test_dims_validated(client) -> None:
    assert client.post("/tick", json={"target": [1.0]}).status_code == 422
    assert client.post("/target", json={"impulse": [1, 2, 3]}).status_code == 422


def test_trajectory_replay(client) -> None:
    assert client.get("/trajectory").json()["frames"] == []
    client.post("/tick", json={"steps": 20, "target": [0.5, 0.5]})
    body = client.get("/trajectory", params={"seconds": 1.0, "rate_hz": 20}).json()
    assert body["end_tick"] == 20 and body["start_tick"] == 10
    assert len(body["samples"]) == 11
    assert len(body["frames"]) == 21


def test_reset(client) -> None:
    client.post("/tick", json={"steps": 5, "target": [1.0, 1.0]})
    body = client.post("/reset", json={}).json()
    assert body["position"] == [0.0, 0.0] and body["tick"] == 5
    client.post("/reset", json={"clear_trajectory": True})
    assert client.get("/trajectory").json()["samples"] == []


def test_affect_override_and_release(client) -> None:
    r = client.post("/affect", json={"valence": -0.8, "arousal": 0.95}).json()
    assert r["affect_source"] == "manual"
    anxious = client.post("/tick", json={"steps": 1}).json()["params"]
    assert anxious["jitter"] > 0
    assert client.post("/affect", json={"valence": 2.0}).status_code == 422
    assert client.post("/affect", json={"release": True}).json()["affect"]["valence"] == pytest.approx(0.1)


def test_providers_hide_token(client) -> None:
    body = client.get("/providers").json()
    assert "chronosoul_token" not in body and body["affect_source"] == "simulated"


def test_background_loop_ticks_without_requests(tmp_path) -> None:
    sub = ContinuousMotionSubstrate(str(tmp_path / "bg.db"), affect_source=StaticAffectSource(), catch_up=False)
    with TestClient(create_app(sub, ProviderSettings(tmp_path / "p.json"), autostart=True)) as c:
        client_target = c.post("/target", json={"target": [1.0, 0.0]}).json()
        assert client_target["queued"]
        time.sleep(0.6)
        state = c.get("/state").json()
    assert state["running"]
    assert 3 <= state["tick"] <= 9
    assert state["position"][0] > 0.05
    sub.store.close()
