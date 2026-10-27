"""Momentum and anticipation: act ahead of the input instead of only reacting to it.

Two predictions, both cheap enough to run every tick:

1. Where the *input* is going. `InputVelocityEstimator` smooths the velocity of the
   target stream. `Anticipator.lead_target` extrapolates the target by
   `momentum_gain * lead_time * input_velocity`, so the spring chases where the input
   will be rather than where it was. A pure spring tracking a ramp of speed u lags
   by about c * u / k; leading the target by that much removes most of the lag.

2. Where the *state* is going. `predict_state` rolls the same spring forward
   `horizon` seconds without noise. With jitter off, a one-tick prediction is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from chronosoul_bridge import SpringParams
from dynamics import Vector, acceleration, integrate


@dataclass(frozen=True)
class Prediction:
    horizon: float
    position: Vector
    velocity: Vector

    def to_dict(self) -> dict[str, object]:
        return {"horizon": self.horizon, "position": self.position, "velocity": self.velocity}


class InputVelocityEstimator:
    """Exponentially smoothed velocity of the input (target) stream."""

    def __init__(self, dims: int, smoothing_time: float = 0.3) -> None:
        self.dims = dims
        self.smoothing_time = smoothing_time
        self.velocity: Vector = [0.0] * dims
        self._last: Optional[Vector] = None

    def reset(self) -> None:
        self.velocity = [0.0] * self.dims
        self._last = None

    def observe(self, target: Sequence[float], dt: float) -> Vector:
        if self._last is None or dt <= 0:
            self._last = list(target)
            return self.velocity
        alpha = 1.0 - math.exp(-dt / self.smoothing_time) if self.smoothing_time > 0 else 1.0
        raw = [(t - p) / dt for t, p in zip(target, self._last)]
        self.velocity = [v + alpha * (r - v) for v, r in zip(self.velocity, raw)]
        self._last = list(target)
        return self.velocity


class Anticipator:
    def __init__(self, dims: int, smoothing_time: float = 0.3, enabled: bool = True) -> None:
        self.enabled = enabled
        self.estimator = InputVelocityEstimator(dims, smoothing_time)

    def reset(self) -> None:
        self.estimator.reset()

    def observe_input(self, target: Sequence[float], dt: float) -> Vector:
        return self.estimator.observe(target, dt)

    @staticmethod
    def lead_time(params: SpringParams) -> float:
        """Seconds a spring of these params lags a constant-speed input: c / k."""
        return params.damping / params.stiffness

    def lead_target(self, target: Sequence[float], params: SpringParams) -> Vector:
        if not self.enabled:
            return list(target)
        lead = params.momentum_gain * self.lead_time(params)
        return [t + lead * v for t, v in zip(target, self.estimator.velocity)]


def predict_state(position: Sequence[float], velocity: Sequence[float], target: Sequence[float],
                  params: SpringParams, horizon: float, dt: float, substeps: int = 4) -> Prediction:
    """Roll the noise-free spring forward `horizon` seconds in ticks of `dt`."""
    x, v = list(position), list(velocity)
    steps = max(0, int(round(horizon / dt)))
    for _ in range(steps):
        x, v = integrate(x, v, target, params, dt, substeps)
    return Prediction(steps * dt, x, v)


def kinematic_extrapolate(position: Sequence[float], velocity: Sequence[float], target: Sequence[float],
                          params: SpringParams, horizon: float) -> Prediction:
    """Second-order Taylor guess x + v*h + a*h^2/2; cheaper but drifts for long horizons."""
    a = acceleration(position, velocity, target, params)
    h = horizon
    pos = [x + v * h + 0.5 * ai * h * h for x, v, ai in zip(position, velocity, a)]
    vel = [v + ai * h for v, ai in zip(velocity, a)]
    return Prediction(h, pos, vel)


def momentum(velocity: Sequence[float], mass: float) -> Vector:
    return [mass * v for v in velocity]
