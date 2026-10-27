"""ChronoSoul bridge: the emotional substrate as an input to the motion substrate.

ChronoSoul (github.com/AaronGrace978/continuous-emotional-substrate) keeps a
continuous affect vector. This module reads valence and arousal from it and turns
them into spring parameters, so mood is felt as motion:

    a' = clamp((arousal - 0.4) / 0.4, -1, 1)     # same mapping as ChronoSoul steering.py
    v  = clamp(valence, -1, 1)

    stiffness  k = k0 * 2 ** a'                         # arousal tightens the spring
    mass       m = m0 * 2 ** (-0.7 * a')                # calm is heavier, aroused is lighter
    damping    z = clamp(z0 * 2 ** (-0.8 * a') * (1 + 0.25 * v), 0.15, 2.5)
    momentum   g = g0 * (1 + 0.5 * a')                  # how far ahead anticipation leads
    jitter     s = s0 * max(0, a') * (1 - v) / 2        # only aroused AND unpleasant shakes

`z` is the damping ratio; the damping coefficient is c = 2 * z * sqrt(k * m).
Anxious (low valence, high arousal) gives a stiff, light, underdamped, jittery
spring; calm (pleasant, low arousal) gives a soft, heavy, overdamped one.

Every source is optional. With no ChronoSoul installed or running, the
`SimulatedAffectSource` drifts with ChronoSoul's own homeostatic constants, so
the repo runs fully offline.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol, runtime_checkable

AROUSAL_REST = 0.4
AROUSAL_SPAN = 0.4


@dataclass(frozen=True)
class AffectReading:
    """The two ChronoSoul coordinates the motion substrate consumes."""

    valence: float = 0.1   # -1.0 (unpleasant) to +1.0 (pleasant)
    arousal: float = 0.25  # 0.0 (quiescent) to 1.0 (frenetic)
    source: str = "default"

    def clamped(self) -> "AffectReading":
        return AffectReading(
            valence=max(-1.0, min(1.0, float(self.valence))),
            arousal=max(0.0, min(1.0, float(self.arousal))),
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class AffectSource(Protocol):
    """Anything that can report the current valence/arousal. Must not block for long."""

    name: str

    def read_affect(self) -> AffectReading: ...


# --------------------------------------------------------------------------- coupling

@dataclass(frozen=True)
class SpringParams:
    """Effective per-tick dynamics, after emotional modulation."""

    stiffness: float
    mass: float
    damping_ratio: float
    momentum_gain: float
    jitter: float

    @property
    def damping(self) -> float:
        return 2.0 * self.damping_ratio * math.sqrt(self.stiffness * self.mass)

    @property
    def natural_frequency(self) -> float:
        return math.sqrt(self.stiffness / self.mass)

    def to_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["damping"] = self.damping
        out["natural_frequency"] = self.natural_frequency
        return out


@dataclass(frozen=True)
class BaseDynamics:
    """The un-modulated spring: what motion looks like at a neutral mood (a' = 0, v = 0)."""

    stiffness: float = 12.0
    mass: float = 1.0
    damping_ratio: float = 0.9
    momentum_gain: float = 1.0
    jitter: float = 0.04


def arousal_coefficient(arousal: float, rest: float = AROUSAL_REST, span: float = AROUSAL_SPAN) -> float:
    """Arousal is stored in [0, 1]; map it so `rest` is 0 and rest +- span is +-1 (as in steering.py)."""
    return max(-1.0, min(1.0, (float(arousal) - rest) / span))


def valence_coefficient(valence: float) -> float:
    return max(-1.0, min(1.0, float(valence)))


def modulate(base: BaseDynamics, affect: AffectReading) -> SpringParams:
    """Map an affect reading onto spring parameters. See the module docstring for the formula."""
    a = arousal_coefficient(affect.arousal)
    v = valence_coefficient(affect.valence)
    damping_ratio = base.damping_ratio * 2.0 ** (-0.8 * a) * (1.0 + 0.25 * v)
    return SpringParams(
        stiffness=base.stiffness * 2.0 ** a,
        mass=base.mass * 2.0 ** (-0.7 * a),
        damping_ratio=max(0.15, min(2.5, damping_ratio)),
        momentum_gain=base.momentum_gain * (1.0 + 0.5 * a),
        jitter=base.jitter * max(0.0, a) * (1.0 - v) / 2.0,
    )


# --------------------------------------------------------------------------- sources

class StaticAffectSource:
    """A fixed mood. Used by tests and the visualizer's manual sliders."""

    def __init__(self, valence: float = 0.1, arousal: float = 0.25, name: str = "manual") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._reading = AffectReading(valence, arousal, name).clamped()

    def set(self, valence: float, arousal: float) -> None:
        with self._lock:
            self._reading = AffectReading(valence, arousal, self.name).clamped()

    def read_affect(self) -> AffectReading:
        with self._lock:
            return self._reading


class SimulatedAffectSource:
    """Offline stand-in for ChronoSoul.

    Uses ChronoSoul's homeostatic drift (valence -> 0.1 at 0.02/s, arousal -> 0.25 at
    0.03/s) plus a slow deterministic wander so the motion has something to feel.
    It integrates on read, from the wall clock, so it costs nothing between reads.
    """

    VALENCE_BASELINE = 0.1
    AROUSAL_BASELINE = 0.25
    VALENCE_RATE = 0.02
    AROUSAL_RATE = 0.03

    def __init__(self, valence: float = 0.1, arousal: float = 0.25, wander: float = 0.25,
                 clock: Any = time.monotonic) -> None:
        self.name = "simulated"
        self.wander = wander
        self._clock = clock
        self._lock = threading.Lock()
        self._valence = valence
        self._arousal = arousal
        self._t0 = clock()
        self._last = self._t0

    def stimulate(self, valence_delta: float = 0.0, arousal_delta: float = 0.0) -> None:
        with self._lock:
            self._valence = max(-1.0, min(1.0, self._valence + valence_delta))
            self._arousal = max(0.0, min(1.0, self._arousal + arousal_delta))

    def read_affect(self) -> AffectReading:
        with self._lock:
            now = self._clock()
            dt = max(0.0, now - self._last)
            self._last = now
            self._valence += (self.VALENCE_BASELINE - self._valence) * min(1.0, self.VALENCE_RATE * dt)
            self._arousal += (self.AROUSAL_BASELINE - self._arousal) * min(1.0, self.AROUSAL_RATE * dt)
            t = now - self._t0
            valence = self._valence + self.wander * math.sin(t / 23.0)
            arousal = self._arousal + self.wander * math.sin(t / 17.0 + 1.3)
        return AffectReading(valence, arousal, self.name).clamped()


class ChronoSoulAffectSource:
    """Reads an in-process ChronoSoul `ContinuousSubstrate` (anything with `.affect.valence/.arousal`)."""

    def __init__(self, substrate: Any) -> None:
        if not hasattr(substrate, "affect"):
            raise TypeError("expected a ChronoSoul substrate with an `.affect` attribute")
        self.name = "chronosoul"
        self.substrate = substrate

    def read_affect(self) -> AffectReading:
        affect = self.substrate.affect
        return AffectReading(float(affect.valence), float(affect.arousal), self.name).clamped()


class ChronoSoulHTTPSource:
    """Polls a running ChronoSoul chamber (`chat_server.py`, GET /state -> {"affect": {...}}).

    Polling happens on a background thread, so `read_affect` never blocks the tick.
    If the chamber is unreachable, the fallback source answers instead.
    """

    def __init__(self, url: str = "http://127.0.0.1:47321/state", token: Optional[str] = None,
                 poll_interval: float = 1.0, timeout: float = 0.5,
                 fallback: Optional[AffectSource] = None) -> None:
        self.name = "chronosoul_http"
        self.url = url
        self._token = token
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.fallback: AffectSource = fallback or SimulatedAffectSource()
        self.last_error: Optional[str] = None
        self._latest: Optional[AffectReading] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def fetch_once(self) -> Optional[AffectReading]:
        request = urllib.request.Request(self.url, headers={"Accept": "application/json"})
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            affect = payload.get("affect", payload)
            reading = AffectReading(float(affect["valence"]), float(affect["arousal"]), self.name).clamped()
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._latest = None
            return None
        self.last_error = None
        with self._lock:
            self._latest = reading
        return reading

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.fetch_once()
            self._stop.wait(self.poll_interval)

    def read_affect(self) -> AffectReading:
        with self._lock:
            latest = self._latest
        return latest if latest is not None else self.fallback.read_affect()


def load_chronosoul_substrate(repo_path: str, state_file: Optional[str] = None) -> Any:
    """Import ChronoSoul's `ContinuousSubstrate` from a local checkout and start its loop.

    Raises ImportError when the checkout or its dependencies are missing; callers
    should fall back to `SimulatedAffectSource`.
    """
    import importlib
    import sys

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    module = importlib.import_module("substrate")
    kwargs = {"state_file": state_file} if state_file else {}
    substrate = module.ContinuousSubstrate(**kwargs)
    substrate.start_background()
    return substrate
