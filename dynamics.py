"""The damped spring, as pure functions shared by the substrate and the predictor.

Per dimension, with stiffness k, mass m, damping c = 2 * z * sqrt(k * m):

    m * x'' = k * (target - x) - c * x' + k * jitter * noise

integrated with semi-implicit (symplectic) Euler over `substeps` per tick:

    v <- v + dt * (k * (target - x) - c * v) / m
    x <- x + dt * v
"""

from __future__ import annotations

from typing import Optional, Sequence

from chronosoul_bridge import SpringParams

Vector = list[float]


def integrate(position: Sequence[float], velocity: Sequence[float], target: Sequence[float],
              params: SpringParams, dt: float, substeps: int = 4,
              noise: Optional[Sequence[float]] = None) -> tuple[Vector, Vector]:
    """Advance one tick of length `dt`. `noise` is a per-dimension unit-variance draw (or None)."""
    x = list(position)
    v = list(velocity)
    k, m, c = params.stiffness, params.mass, params.damping
    h = dt / substeps
    kick = [0.0] * len(x)
    if noise is not None and params.jitter > 0.0:
        # Noise is drawn once per tick and spread over substeps, so it scales with sqrt(dt)
        # like a Brownian force and doesn't depend on the substep count.
        kick = [k * params.jitter * n / (dt ** 0.5) for n in noise]
    for _ in range(substeps):
        for i in range(len(x)):
            accel = (k * (target[i] - x[i]) - c * v[i] + kick[i]) / m
            v[i] += h * accel
            x[i] += h * v[i]
    return x, v


def acceleration(position: Sequence[float], velocity: Sequence[float], target: Sequence[float],
                 params: SpringParams) -> Vector:
    k, m, c = params.stiffness, params.mass, params.damping
    return [(k * (t - x) - c * v) / m for x, v, t in zip(position, velocity, target)]


def decay_toward(value: Sequence[float], rest: Sequence[float], dt: float, half_life: float) -> Vector:
    """Exponential decay of `value` toward `rest` with the given half-life (ChronoSoul's e^(-lambda t))."""
    if half_life <= 0:
        return list(rest)
    keep = 0.5 ** (dt / half_life)
    return [r + (x - r) * keep for x, r in zip(value, rest)]
