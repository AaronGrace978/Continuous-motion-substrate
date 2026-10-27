"""FastAPI server for the motion substrate, with a live canvas visualizer at `/`.

    python server.py                 # http://127.0.0.1:47400
    MOTION_PORT=8000 python server.py

Endpoints:
    GET  /            live visualizer
    GET  /state       current state, affect, spring params, 0.5 s prediction
    POST /tick        advance N ticks synchronously (optionally set a target first)
    POST /target      queue a new target (or an impulse) for the running loop
    GET  /trajectory  recent path: raw samples plus a Hermite-interpolated replay
    POST /reset       return to rest (optionally clear the stored trajectory)
    POST /affect      override valence/arousal by hand, or release back to the configured source
    GET  /providers   affect-source settings (the token is never returned)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chronosoul_bridge import AffectSource, StaticAffectSource
from motion_substrate import ContinuousMotionSubstrate
from providers import ProviderSettings
from trajectory import replay

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("MOTION_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOTION_PORT", "47400"))
DB_PATH = os.environ.get("MOTION_DB", str(ROOT / "motion_substrate.db"))
SETTINGS_PATH = ROOT / "motion_providers.json"
VISUALIZER = ROOT / "static" / "index.html"


class TickRequest(BaseModel):
    steps: int = Field(1, ge=1, le=10_000)
    target: Optional[list[float]] = None


class TargetRequest(BaseModel):
    target: Optional[list[float]] = None
    impulse: Optional[list[float]] = None


class ResetRequest(BaseModel):
    clear_trajectory: bool = False


class AffectRequest(BaseModel):
    valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    release: bool = False


class MotionService:
    """Holds the substrate and the configured vs. manually overridden affect source."""

    def __init__(self, substrate: ContinuousMotionSubstrate, settings: ProviderSettings) -> None:
        self.substrate = substrate
        self.settings = settings
        self.configured: AffectSource = substrate.affect_source
        self.manual = StaticAffectSource()

    def override_affect(self, valence: Optional[float], arousal: Optional[float]) -> None:
        current = self.substrate.affect_source.read_affect()
        self.manual.set(current.valence if valence is None else valence,
                        current.arousal if arousal is None else arousal)
        self.substrate.affect_source = self.manual

    def release_affect(self) -> None:
        self.substrate.affect_source = self.configured


def create_app(substrate: Optional[ContinuousMotionSubstrate] = None,
               settings: Optional[ProviderSettings] = None,
               autostart: bool = True) -> FastAPI:
    settings = settings or ProviderSettings(SETTINGS_PATH)
    substrate = substrate or ContinuousMotionSubstrate(DB_PATH, affect_source=settings.build_source())
    service = MotionService(substrate, settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if autostart:
            substrate.start_background()
        try:
            yield
        finally:
            substrate.stop_background()

    app = FastAPI(title="Continuous Motion Substrate", lifespan=lifespan)
    app.state.service = service

    def check_dims(values: Optional[list[float]]) -> None:
        if values is not None and len(values) != substrate.dims:
            raise HTTPException(422, f"expected {substrate.dims} values, got {len(values)}")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return VISUALIZER.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "running": substrate.running, "tick": substrate.state.tick}

    @app.get("/state")
    def state() -> dict[str, object]:
        snap = substrate.snapshot()
        snap["affect_source"] = substrate.affect_source.name
        return snap

    @app.post("/tick")
    def tick(req: TickRequest) -> dict[str, object]:
        check_dims(req.target)
        if req.target is not None:
            substrate.set_target(req.target)
        substrate.run_ticks(req.steps)
        return substrate.snapshot()

    @app.post("/target")
    def target(req: TargetRequest) -> dict[str, bool]:
        check_dims(req.target)
        check_dims(req.impulse)
        if req.target is not None:
            substrate.set_target(req.target)
        if req.impulse is not None:
            substrate.nudge(req.impulse)
        return {"queued": req.target is not None or req.impulse is not None}

    @app.get("/trajectory")
    def trajectory(seconds: float = Query(5.0, gt=0, le=3600),
                   rate_hz: float = Query(30.0, gt=0, le=240),
                   include_samples: bool = True) -> dict[str, object]:
        last = substrate.store.last_tick()
        if last is None:
            return {"samples": [], "frames": [], "start_tick": None, "end_tick": None}
        start = max(0, last - int(seconds * substrate.tick_hz))
        samples = substrate.store.window(start, last)
        frames = [f.to_dict() for f in replay(samples, rate_hz)]
        return {
            "start_tick": start,
            "end_tick": last,
            "rate_hz": rate_hz,
            "samples": [s.to_dict() for s in samples] if include_samples else [],
            "frames": frames,
        }

    @app.post("/reset")
    def reset(req: Optional[ResetRequest] = None) -> dict[str, object]:
        substrate.reset(clear_trajectory=bool(req and req.clear_trajectory))
        return substrate.snapshot()

    @app.post("/affect")
    def affect(req: AffectRequest) -> dict[str, object]:
        if req.release:
            service.release_affect()
        else:
            service.override_affect(req.valence, req.arousal)
        return {"affect_source": substrate.affect_source.name,
                "affect": substrate.affect_source.read_affect().to_dict()}

    @app.get("/providers")
    def providers() -> dict[str, object]:
        return service.settings.public()

    return app


if __name__ == "__main__":
    import uvicorn

    print(f"Continuous Motion Substrate on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="warning")
