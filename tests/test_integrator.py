from __future__ import annotations

import math

import pytest

from chronosoul_bridge import SpringParams
from dynamics import integrate


def analytic_underdamped(t: float, k: float, m: float, z: float) -> float:
    """Unit step response from rest of m x'' + c x' + k x = k."""
    w = math.sqrt(k / m)
    wd = w * math.sqrt(1 - z * z)
    return 1 - math.exp(-z * w * t) * (math.cos(wd * t) + z / math.sqrt(1 - z * z) * math.sin(wd * t))


def test_converges_to_target(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([1.0, -0.5])
    result = sub.run_ticks(100, record=False)
    assert result.position == pytest.approx([1.0, -0.5], abs=1e-4)
    assert result.velocity == pytest.approx([0.0, 0.0], abs=1e-3)


def test_matches_analytic_damped_spring() -> None:
    params = SpringParams(stiffness=12.0, mass=1.0, damping_ratio=0.4, momentum_gain=1.0, jitter=0.0)
    x, v = [0.0], [0.0]
    dt, worst = 0.1, 0.0
    for i in range(1, 60):
        x, v = integrate(x, v, [1.0], params, dt, substeps=4)
        worst = max(worst, abs(x[0] - analytic_underdamped(i * dt, 12.0, 1.0, 0.4)))
    assert worst < 0.05


def test_more_substeps_is_more_accurate() -> None:
    params = SpringParams(stiffness=12.0, mass=1.0, damping_ratio=0.4, momentum_gain=1.0, jitter=0.0)

    def error(substeps: int) -> float:
        x, v, worst = [0.0], [0.0], 0.0
        for i in range(1, 60):
            x, v = integrate(x, v, [1.0], params, 0.1, substeps)
            worst = max(worst, abs(x[0] - analytic_underdamped(i * 0.1, 12.0, 1.0, 0.4)))
        return worst

    assert error(16) < error(4) < error(1)


def test_fixed_tick_is_deterministic(make_substrate) -> None:
    a = make_substrate(valence=-0.8, arousal=0.95, jitter=True, db="a.db", seed=3)
    b = make_substrate(valence=-0.8, arousal=0.95, jitter=True, db="b.db", seed=3)
    for sub in (a, b):
        sub.set_target([0.7, 0.2])
    for _ in range(80):
        ra, rb = a.tick_once(), b.tick_once()
        assert ra.position == rb.position


def test_different_seed_gives_different_jitter(make_substrate) -> None:
    a = make_substrate(valence=-0.8, arousal=0.95, jitter=True, db="a.db", seed=1)
    b = make_substrate(valence=-0.8, arousal=0.95, jitter=True, db="b.db", seed=2)
    assert a.run_ticks(20).position != b.run_ticks(20).position


def test_impulse_carries_momentum(make_substrate) -> None:
    sub = make_substrate()
    sub.nudge([2.0, 0.0])
    first = sub.tick_once()
    assert first.position[0] > 0.1
    assert sub.snapshot()["momentum"][0] == pytest.approx(first.velocity[0] * first.params.mass)
    sub.run_ticks(100)
    assert sub.state.position[0] == pytest.approx(0.0, abs=1e-3)


def test_state_keeps_moving_between_inputs(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([1.0, 0.0])
    positions = [sub.tick_once().position[0] for _ in range(5)]
    assert all(b > a for a, b in zip(positions, positions[1:]))


def test_wrong_dimension_rejected(make_substrate) -> None:
    with pytest.raises(ValueError):
        make_substrate().set_target([1.0])
