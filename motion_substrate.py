"""The continuous motion substrate: a state vector that keeps moving between turns.

One class, one explicit tick loop, as in ChronoSoul's `ContinuousSubstrate`.
Every tick (10 Hz by default, fixed dt):

  1. drain the inbox (new targets, impulses)
  2. read valence/arousal from the affect source and modulate the spring
  3. decay the target toward rest (homeostasis: with no input, motion settles)
  4. estimate the input's velocity and lead the target (momentum.py)
  5. integrate the damped spring (dynamics.py), with mood-dependent jitter
  6. append the sample to the trajectory store (trajectory.py)

State lives in SQLite (`motion_substrate.db`): the trajectory in a hot/cold
sample table, the latest state vector in `meta`. On restart the substrate
reloads and, like ChronoSoul's dormancy, catches up on the time it was closed.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from chronosoul_bridge import (
    AffectReading,
    AffectSource,
    BaseDynamics,
    SimulatedAffectSource,
    SpringParams,
    modulate,
)
from dynamics import Vector, decay_toward, integrate
from momentum import Anticipator, momentum, predict_state
from trajectory import Sample, TrajectoryStore

STATE_KEY = "motion_state"
STATE_VERSION = 1


@dataclass
class MotionState:
    position: Vector
    velocity: Vector
    target: Vector
    rest: Vector
    tick: int = 0
    sim_time: float = 0.0

    @classmethod
    def at_rest(cls, dims: int, rest: Optional[Sequence[float]] = None) -> "MotionState":
        r = list(rest) if rest is not None else [0.0] * dims
        return cls(position=list(r), velocity=[0.0] * dims, target=list(r), rest=list(r))


@dataclass
class TickResult:
    tick: int
    sim_time: float
    position: Vector
    velocity: Vector
    target: Vector
    driven_target: Vector
    affect: AffectReading
    params: SpringParams
    extra: dict[str, Any] = field(default_factory=dict)


class ContinuousMotionSubstrate:
    TICK_HZ = 10.0
    SUBSTEPS = 4
    TARGET_HALF_LIFE = 8.0     # seconds for an unrefreshed target to fall halfway back to rest
    MAX_CATCHUP_SEC = 600.0    # dormancy catch-up is capped; past this the spring has long settled
    SAVE_EVERY = 50            # ticks between state flushes
    MAX_BEHIND_TICKS = 5       # background loop: ticks run at once when late before dropping
    PREDICT_HORIZON = 0.5      # seconds ahead reported by snapshot()

    def __init__(self,
                 db_file: str = "motion_substrate.db",
                 dims: int = 2,
                 tick_hz: float = TICK_HZ,
                 affect_source: Optional[AffectSource] = None,
                 base: BaseDynamics = BaseDynamics(),
                 seed: int = 0,
                 anticipate: bool = True,
                 catch_up: bool = True,
                 hot_limit: int = 6000,
                 target_half_life: Optional[float] = None,
                 clock: Any = time.time) -> None:
        if dims < 1:
            raise ValueError("dims must be >= 1")
        self.db_file = db_file
        self.dims = dims
        self.tick_hz = tick_hz
        self.dt = 1.0 / tick_hz
        self.affect_source: AffectSource = affect_source or SimulatedAffectSource()
        self.base = base
        self.seed = seed
        self.target_half_life = self.TARGET_HALF_LIFE if target_half_life is None else target_half_life
        self._clock = clock
        self.store = TrajectoryStore(db_file, hot_limit=hot_limit)
        self.anticipator = Anticipator(dims, enabled=anticipate)
        self.state = MotionState.at_rest(dims)
        self.inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self.last_affect = AffectReading()
        self.last_params = modulate(base, self.last_affect)
        self.last_driven: Vector = list(self.state.target)
        self.saved_at: Optional[float] = None
        self.dormant_for = 0.0
        self.dropped_ticks = 0
        self.running = False
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self.load_state(catch_up=catch_up)

    # ------------------------------------------------------------------ input
    def set_target(self, target: Sequence[float]) -> None:
        """Queue a new target. Takes effect on the next tick."""
        self.inbox.put({"type": "target", "value": self._vec(target)})

    def nudge(self, impulse: Sequence[float]) -> None:
        """Queue a velocity impulse (an external push)."""
        self.inbox.put({"type": "impulse", "value": self._vec(impulse)})

    def set_rest(self, rest: Sequence[float]) -> None:
        self.inbox.put({"type": "rest", "value": self._vec(rest)})

    def _vec(self, values: Sequence[float]) -> Vector:
        out = [float(x) for x in values]
        if len(out) != self.dims:
            raise ValueError(f"expected {self.dims} values, got {len(out)}")
        return out

    def _drain_inbox(self) -> None:
        while True:
            try:
                event = self.inbox.get_nowait()
            except queue.Empty:
                return
            if event["type"] == "target":
                self.state.target = event["value"]
            elif event["type"] == "impulse":
                self.state.velocity = [v + dv for v, dv in zip(self.state.velocity, event["value"])]
            elif event["type"] == "rest":
                self.state.rest = event["value"]

    # ------------------------------------------------------------------ the tick
    def _noise(self, tick: int) -> list[float]:
        rng = random.Random(self.seed * 1_000_003 + tick)
        return [rng.gauss(0.0, 1.0) for _ in range(self.dims)]

    def tick_once(self, record: bool = True) -> TickResult:
        with self._lock:
            self._drain_inbox()
            s = self.state
            dt = self.dt

            affect = self.affect_source.read_affect()
            params = modulate(self.base, affect)

            if self.target_half_life != float("inf"):
                s.target = decay_toward(s.target, s.rest, dt, self.target_half_life)
            self.anticipator.observe_input(s.target, dt)
            driven = self.anticipator.lead_target(s.target, params)

            noise = self._noise(s.tick) if params.jitter > 0 else None
            s.position, s.velocity = integrate(s.position, s.velocity, driven, params, dt, self.SUBSTEPS, noise)
            s.tick += 1
            s.sim_time = s.tick * dt

            self.last_affect, self.last_params, self.last_driven = affect, params, driven
            if record:
                self.store.append(Sample(
                    tick=s.tick, sim_time=s.sim_time, wall_time=self._clock(),
                    position=list(s.position), velocity=list(s.velocity), target=list(s.target),
                    valence=affect.valence, arousal=affect.arousal,
                ))
                if s.tick % self.SAVE_EVERY == 0:
                    self.save_state()
            return TickResult(s.tick, s.sim_time, list(s.position), list(s.velocity),
                              list(s.target), list(driven), affect, params)

    def run_ticks(self, n: int, record: bool = True) -> TickResult:
        if n < 1:
            raise ValueError("n must be >= 1")
        result = self.tick_once(record)
        for _ in range(n - 1):
            result = self.tick_once(record)
        return result

    # ------------------------------------------------------------------ background loop
    def _run_loop(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            behind = int((now - next_at) / self.dt)
            if behind > self.MAX_BEHIND_TICKS:
                self.dropped_ticks += behind - self.MAX_BEHIND_TICKS
                next_at += (behind - self.MAX_BEHIND_TICKS) * self.dt
            while next_at <= time.monotonic():
                self.tick_once()
                next_at += self.dt
            self._stop.wait(max(0.0, next_at - time.monotonic()))

    def start_background(self) -> None:
        if self.running:
            return
        self.running = True
        self._stop.clear()
        self._worker = threading.Thread(target=self._run_loop, name="motion-substrate", daemon=True)
        self._worker.start()

    def stop_background(self) -> None:
        self.running = False
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.save_state()

    def close(self) -> None:
        self.stop_background()
        self.store.close()

    # ------------------------------------------------------------------ views
    def predict(self, horizon: float = PREDICT_HORIZON) -> dict[str, Any]:
        with self._lock:
            s = self.state
            return predict_state(s.position, s.velocity, self.last_driven, self.last_params,
                                 horizon, self.dt, self.SUBSTEPS).to_dict()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            s = self.state
            speed = sum(v * v for v in s.velocity) ** 0.5
            return {
                "tick": s.tick,
                "sim_time": s.sim_time,
                "tick_hz": self.tick_hz,
                "dims": self.dims,
                "position": list(s.position),
                "velocity": list(s.velocity),
                "speed": speed,
                "momentum": momentum(s.velocity, self.last_params.mass),
                "target": list(s.target),
                "driven_target": list(self.last_driven),
                "rest": list(s.rest),
                "affect": self.last_affect.to_dict(),
                "params": self.last_params.to_dict(),
                "prediction": self.predict(),
                "anticipate": self.anticipator.enabled,
                "input_velocity": list(self.anticipator.estimator.velocity),
                "running": self.running,
                "dormant_for": self.dormant_for,
                "dropped_ticks": self.dropped_ticks,
                "trajectory": {"hot": self.store.count("hot"), "cold": self.store.count("cold")},
            }

    # ------------------------------------------------------------------ reset
    def reset(self, clear_trajectory: bool = False) -> None:
        """Return to rest. The tick counter keeps counting so stored history stays unambiguous."""
        with self._lock:
            while not self.inbox.empty():
                self.inbox.get_nowait()
            tick, sim_time = self.state.tick, self.state.sim_time
            self.state = MotionState.at_rest(self.dims, self.state.rest)
            self.state.tick, self.state.sim_time = tick, sim_time
            self.anticipator.reset()
            self.last_driven = list(self.state.target)
            if clear_trajectory:
                self.store.clear()
            self.save_state()

    # ------------------------------------------------------------------ persistence
    def save_state(self) -> None:
        with self._lock:
            est = self.anticipator.estimator
            payload = {
                "version": STATE_VERSION,
                "saved_at": self._clock(),
                "dims": self.dims,
                "state": asdict(self.state),
                "input_velocity": est.velocity,
                "input_last": est._last,
            }
            self.store.set_meta(STATE_KEY, json.dumps(payload))
            self.store.commit()

    def load_state(self, catch_up: bool = True) -> None:
        raw = self.store.get_meta(STATE_KEY)
        if not raw:
            return
        try:
            data = json.loads(raw)
            if data.get("dims") != self.dims:
                print(f"[MotionSubstrate] Saved state has dims={data.get('dims')}, expected {self.dims}; starting at rest")
                return
            self.state = MotionState(**data["state"])
            self.anticipator.estimator.velocity = list(data.get("input_velocity") or [0.0] * self.dims)
            last = data.get("input_last")
            self.anticipator.estimator._last = list(last) if last is not None else None
            self.last_driven = list(self.state.target)
            self.saved_at = float(data.get("saved_at") or 0.0)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[MotionSubstrate] Warning loading state: {exc}")
            return
        if catch_up and self.saved_at:
            self.catch_up(self._clock() - self.saved_at)

    def catch_up(self, elapsed: float) -> int:
        """Integrate the time the process was closed. Only the final sample is recorded."""
        elapsed = max(0.0, elapsed)
        self.dormant_for = elapsed
        steps = int(min(elapsed, self.MAX_CATCHUP_SEC) / self.dt)
        if steps <= 0:
            return 0
        for _ in range(steps - 1):
            self.tick_once(record=False)
        self.tick_once(record=True)
        self.save_state()
        return steps
