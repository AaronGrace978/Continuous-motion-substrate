from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from chronosoul_bridge import (
    AffectSource,
    ChronoSoulAffectSource,
    ChronoSoulHTTPSource,
    SimulatedAffectSource,
    StaticAffectSource,
)
from motion_substrate import ContinuousMotionSubstrate


@dataclass
class FakeAffectiveState:
    """The shape of ChronoSoul's `AffectiveState` (substrate.py)."""

    valence: float = 0.2
    arousal: float = 0.3
    dominance: float = 0.5
    curiosity: float = 0.7
    solitude: float = 0.1
    entropy: float = 0.05


class FakeChronoSoul:
    def __init__(self) -> None:
        self.affect = FakeAffectiveState()


def test_sources_satisfy_protocol() -> None:
    for source in (StaticAffectSource(), SimulatedAffectSource(), ChronoSoulAffectSource(FakeChronoSoul()),
                   ChronoSoulHTTPSource("http://127.0.0.1:9/state")):
        assert isinstance(source, AffectSource)


def test_in_process_chronosoul_is_read_live() -> None:
    soul = FakeChronoSoul()
    source = ChronoSoulAffectSource(soul)
    assert source.read_affect().valence == pytest.approx(0.2)
    soul.affect.valence, soul.affect.arousal = -3.0, 0.9
    reading = source.read_affect()
    assert (reading.valence, reading.arousal) == (-1.0, 0.9)
    with pytest.raises(TypeError):
        ChronoSoulAffectSource(object())


def test_chronosoul_drives_the_substrate(tmp_path) -> None:
    soul = FakeChronoSoul()
    sub = ContinuousMotionSubstrate(str(tmp_path / "m.db"), affect_source=ChronoSoulAffectSource(soul), catch_up=False)
    calm = sub.tick_once().params
    soul.affect.valence, soul.affect.arousal = -0.8, 0.95
    anxious = sub.tick_once().params
    assert anxious.stiffness > calm.stiffness and anxious.damping_ratio < calm.damping_ratio
    sub.close()


def test_simulated_source_drifts_to_chronosoul_baseline() -> None:
    t = [0.0]
    source = SimulatedAffectSource(valence=0.9, arousal=0.9, wander=0.0, clock=lambda: t[0])
    for _ in range(600):
        t[0] += 1.0
        reading = source.read_affect()
    assert reading.valence == pytest.approx(SimulatedAffectSource.VALENCE_BASELINE, abs=0.01)
    assert reading.arousal == pytest.approx(SimulatedAffectSource.AROUSAL_BASELINE, abs=0.01)


def test_simulated_stimulus_is_clamped() -> None:
    source = SimulatedAffectSource(wander=0.0, clock=lambda: 0.0)
    source.stimulate(valence_delta=-5.0, arousal_delta=5.0)
    reading = source.read_affect()
    assert (reading.valence, reading.arousal) == (-1.0, 1.0)


@pytest.fixture
def chamber() -> Iterator[tuple[str, list[dict]]]:
    """A stand-in for ChronoSoul's chat_server: GET /state -> {"affect": {...}, "mood": ...}."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append({"path": self.path, "auth": self.headers.get("Authorization")})
            body = json.dumps({"affect": {"valence": -0.6, "arousal": 0.85, "dominance": 0.5},
                               "mood": "heavy and aching"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/state", seen
    server.shutdown()


def test_http_source_reads_chamber(chamber) -> None:
    url, seen = chamber
    source = ChronoSoulHTTPSource(url, token="test-token")
    assert source.fetch_once() is not None
    reading = source.read_affect()
    assert (reading.valence, reading.arousal, reading.source) == (-0.6, 0.85, "chronosoul_http")
    assert seen[0]["auth"] == "Bearer test-token"


def test_http_source_falls_back_when_unreachable() -> None:
    source = ChronoSoulHTTPSource("http://127.0.0.1:9/state", timeout=0.2,
                                  fallback=StaticAffectSource(0.3, 0.2, name="fallback"))
    assert source.fetch_once() is None
    assert source.last_error
    assert source.read_affect().source == "fallback"
