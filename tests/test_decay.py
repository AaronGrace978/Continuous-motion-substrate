from __future__ import annotations

import pytest

from dynamics import decay_toward


def test_decay_toward_half_life() -> None:
    assert decay_toward([1.0], [0.0], dt=2.0, half_life=2.0) == pytest.approx([0.5])
    assert decay_toward([1.0], [0.0], dt=4.0, half_life=2.0) == pytest.approx([0.25])
    assert decay_toward([3.0], [1.0], dt=2.0, half_life=2.0) == pytest.approx([2.0])


def test_decay_is_step_size_independent() -> None:
    one = decay_toward([1.0], [0.0], dt=1.0, half_life=3.0)
    many = [1.0]
    for _ in range(10):
        many = decay_toward(many, [0.0], dt=0.1, half_life=3.0)
    assert many == pytest.approx(one)


def test_unrefreshed_target_decays_to_rest(make_substrate) -> None:
    sub = make_substrate(target_half_life=2.0)
    sub.set_target([1.0, 0.0])
    sub.run_ticks(20, record=False)
    assert sub.state.target[0] == pytest.approx(0.5, rel=1e-9)
    sub.run_ticks(20, record=False)
    assert sub.state.target[0] == pytest.approx(0.25, rel=1e-9)


def test_motion_settles_with_no_input(make_substrate) -> None:
    sub = make_substrate(target_half_life=1.0)
    sub.set_target([1.0, 1.0])
    sub.run_ticks(5)
    assert abs(sub.state.position[0]) > 0.05
    sub.run_ticks(300)
    assert sub.state.position == pytest.approx([0.0, 0.0], abs=1e-3)
    assert sub.snapshot()["speed"] < 1e-3


def test_settles_to_moved_rest(make_substrate) -> None:
    sub = make_substrate(target_half_life=0.5)
    sub.set_rest([0.3, -0.3])
    sub.run_ticks(200)
    assert sub.state.position == pytest.approx([0.3, -0.3], abs=1e-3)
