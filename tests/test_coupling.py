"""The emotional coupling: ChronoSoul's valence/arousal shape the motion."""

from __future__ import annotations

import math

import pytest

import motion_eval
from chronosoul_bridge import AffectReading, BaseDynamics, arousal_coefficient, modulate


def test_neutral_mood_is_the_base_spring() -> None:
    base = BaseDynamics()
    p = modulate(base, AffectReading(valence=0.0, arousal=0.4))
    assert (p.stiffness, p.mass, p.damping_ratio, p.momentum_gain, p.jitter) == (
        base.stiffness, base.mass, base.damping_ratio, base.momentum_gain, 0.0)


def test_formula_at_the_extremes() -> None:
    base = BaseDynamics()
    anxious = modulate(base, AffectReading(valence=-1.0, arousal=0.8))  # a' = +1
    assert anxious.stiffness == pytest.approx(2 * base.stiffness)
    assert anxious.mass == pytest.approx(base.mass * 2 ** -0.7)
    assert anxious.damping_ratio == pytest.approx(base.damping_ratio * 2 ** -0.8 * 0.75)
    assert anxious.momentum_gain == pytest.approx(1.5 * base.momentum_gain)
    assert anxious.jitter == pytest.approx(base.jitter)

    calm = modulate(base, AffectReading(valence=1.0, arousal=0.0))  # a' = -1
    assert calm.stiffness == pytest.approx(base.stiffness / 2)
    assert calm.mass == pytest.approx(base.mass * 2 ** 0.7)
    assert calm.damping_ratio == pytest.approx(min(2.5, base.damping_ratio * 2 ** 0.8 * 1.25))
    assert calm.jitter == 0.0


def test_excited_but_pleasant_does_not_shake() -> None:
    assert modulate(BaseDynamics(), AffectReading(valence=1.0, arousal=1.0)).jitter == 0.0


def test_arousal_mapping_matches_chronosoul_steering() -> None:
    assert arousal_coefficient(0.4) == 0.0
    assert arousal_coefficient(0.0) == -1.0
    assert arousal_coefficient(0.8) == 1.0
    assert arousal_coefficient(1.0) == 1.0


def test_damping_ratio_is_clamped() -> None:
    p = modulate(BaseDynamics(damping_ratio=5.0), AffectReading(valence=1.0, arousal=0.0))
    assert p.damping_ratio == 2.5
    assert p.damping == pytest.approx(2 * 2.5 * math.sqrt(p.stiffness * p.mass))


def test_anxious_overshoots_calm_does_not() -> None:
    anxious, calm = motion_eval.step_response("anxious"), motion_eval.step_response("calm")
    assert anxious["overshoot_pct"] > 10.0
    assert calm["overshoot_pct"] == 0.0


def test_calm_is_slower_and_heavier() -> None:
    rise = {m: motion_eval.step_response(m)["rise_time_s"] for m in motion_eval.MOODS}
    assert rise["anxious"] < rise["neutral"] < rise["calm"]


def test_only_anxious_jitters() -> None:
    assert motion_eval.jitter_rms("anxious") > 0.01
    assert motion_eval.jitter_rms("calm") == 0.0
    assert motion_eval.jitter_rms("neutral") == 0.0


def test_anxious_runs_ahead_calm_trails() -> None:
    assert motion_eval.ramp_lag("anxious", anticipate=True) < 0.0
    assert motion_eval.ramp_lag("calm", anticipate=True) > 0.0


def test_mood_change_mid_motion_takes_effect_next_tick(make_substrate) -> None:
    sub = make_substrate(valence=0.6, arousal=0.0)
    sub.set_target([1.0, 0.0])
    calm_params = sub.tick_once().params
    sub.affect_source.set(-0.8, 0.95)
    anxious_params = sub.tick_once().params
    assert anxious_params.stiffness > calm_params.stiffness
    assert sub.store.latest(1)[0].arousal == pytest.approx(0.95)
