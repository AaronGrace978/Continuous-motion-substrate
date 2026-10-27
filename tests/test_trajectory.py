from __future__ import annotations

import pytest

from trajectory import Sample, hermite, interpolate, path_length, replay


def cubic(t: float) -> tuple[float, float]:
    return 0.5 * t ** 3 - t ** 2 + 0.25 * t + 1.0, 1.5 * t ** 2 - 2 * t + 0.25


def sample(tick: int, t: float, x: float, v: float) -> Sample:
    return Sample(tick=tick, sim_time=t, wall_time=t, position=[x], velocity=[v], target=[0.0])


def test_hermite_hits_endpoints() -> None:
    a, b = sample(0, 0.0, 1.0, 2.0), sample(1, 0.1, 1.3, -1.0)
    assert hermite(a, b, 0.0).position == pytest.approx([1.0])
    assert hermite(a, b, 0.1).position == pytest.approx([1.3])
    assert hermite(a, b, 0.0).velocity == pytest.approx([2.0])
    assert hermite(a, b, 0.1).velocity == pytest.approx([-1.0])


def test_hermite_exact_for_cubics() -> None:
    samples = [sample(i, i * 0.1, *cubic(i * 0.1)) for i in range(11)]
    for t in (0.03, 0.27, 0.55, 0.91):
        frame = interpolate(samples, t)
        assert frame.position == pytest.approx([cubic(t)[0]], abs=1e-12)
        assert frame.velocity == pytest.approx([cubic(t)[1]], abs=1e-12)


def test_interpolate_clamps_to_ends() -> None:
    samples = [sample(1, 0.1, 1.0, 0.0), sample(2, 0.2, 2.0, 0.0)]
    assert interpolate(samples, -5.0).position == [1.0]
    assert interpolate(samples, 5.0).position == [2.0]
    with pytest.raises(ValueError):
        interpolate([], 0.0)


def test_replay_rate_and_identity_at_tick_rate(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([1.0, -1.0])
    sub.run_ticks(30)
    samples = sub.store.window(1, 30)

    at_tick_rate = list(replay(samples, rate_hz=10.0))
    assert len(at_tick_rate) == 30
    for frame, s in zip(at_tick_rate, samples):
        assert frame.position == pytest.approx(s.position, abs=1e-12)

    upsampled = list(replay(samples, rate_hz=60.0))
    assert len(upsampled) == 29 * 6 + 1
    assert upsampled[-1].position == pytest.approx(samples[-1].position)


def test_upsampled_replay_tracks_true_motion(make_substrate) -> None:
    """Hermite replay of 10 Hz samples vs. the true sub-tick states.

    The 40 Hz run with 1 substep takes the exact same 0.025 s integration steps as the
    10 Hz run with 4 substeps, so the difference is interpolation error alone.
    """
    coarse = make_substrate(db="coarse.db")
    fine = make_substrate(db="fine.db", tick_hz=40.0)
    fine.SUBSTEPS = 1
    for sub in (coarse, fine):
        sub.set_target([1.0, 0.0])
    coarse.run_ticks(20)
    fine.run_ticks(80)
    frames = list(replay(coarse.store.window(1, 20), rate_hz=40.0))
    truth = fine.store.window(4, 80)
    worst = max(abs(f.position[0] - s.position[0]) for f, s in zip(frames, truth))
    assert worst < 0.005
    on_ticks = max(abs(f.position[0] - s.position[0]) for f, s in zip(frames[::4], truth[::4]))
    assert on_ticks == pytest.approx(0.0, abs=1e-12)


def test_replay_window_across_tiers(make_substrate) -> None:
    sub = make_substrate(hot_limit=20)
    sub.store.archive_every = 5
    sub.set_target([1.0, 0.0])
    sub.run_ticks(60)
    assert sub.store.count("cold") > 0
    window = sub.store.window(10, 50)
    assert [s.tick for s in window] == list(range(10, 51))
    assert len(list(replay(window, rate_hz=20.0))) == 81


def test_path_length_is_longer_than_displacement_when_overshooting(make_substrate) -> None:
    sub = make_substrate(valence=-0.8, arousal=0.95)
    sub.set_target([1.0, 0.0])
    sub.run_ticks(40)
    assert path_length(sub.store.window(1, 40)) > 1.1


def test_replay_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        list(replay([sample(0, 0.0, 0.0, 0.0)], rate_hz=0.0))
