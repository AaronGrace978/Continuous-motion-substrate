from __future__ import annotations

import sqlite3

import pytest


def test_state_round_trip(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([0.8, -0.4])
    sub.run_ticks(17)
    before = (list(sub.state.position), list(sub.state.velocity), list(sub.state.target), sub.state.tick)
    sub.close()

    again = make_substrate()
    after = (again.state.position, again.state.velocity, again.state.target, again.state.tick)
    assert after == before
    assert again.store.count() == 17


def test_restart_continues_exactly(make_substrate) -> None:
    """A restart mid-motion lands on the same path as a run that never stopped."""
    straight = make_substrate(valence=-0.8, arousal=0.95, jitter=True, anticipate=True, db="straight.db")
    straight.set_target([1.0, 0.5])
    straight.run_ticks(40)
    expected = straight.run_ticks(40).position

    first = make_substrate(valence=-0.8, arousal=0.95, jitter=True, anticipate=True, db="restart.db")
    first.set_target([1.0, 0.5])
    first.run_ticks(40)
    first.close()
    second = make_substrate(valence=-0.8, arousal=0.95, jitter=True, anticipate=True, db="restart.db")
    assert second.run_ticks(40).position == expected


def test_dormancy_catch_up(make_substrate, clock) -> None:
    sub = make_substrate()
    sub.set_target([1.0, 0.0])
    sub.run_ticks(3)
    sub.close()

    clock.advance(30.0)
    again = make_substrate(catch_up=True)
    assert again.dormant_for == pytest.approx(30.0)
    assert again.state.tick == 3 + 300
    assert again.state.position[0] == pytest.approx(1.0, abs=1e-3)
    assert again.store.count() == 4  # dormant ticks are integrated but only the last is recorded


def test_catch_up_is_capped(make_substrate, clock) -> None:
    sub = make_substrate()
    sub.run_ticks(1)
    sub.close()
    clock.advance(10 * 86400)
    again = make_substrate(catch_up=True)
    assert again.state.tick == 1 + int(again.MAX_CATCHUP_SEC * again.tick_hz)


def test_overflow_archived_never_deleted(make_substrate, tmp_path) -> None:
    sub = make_substrate(hot_limit=50)
    sub.store.archive_every = 10
    sub.run_ticks(230)
    sub.store.archive_overflow()
    assert sub.store.count() == 230
    assert sub.store.count("hot") == 50
    assert sub.store.count("cold") == 180
    window = sub.store.window(1, 230)
    assert [s.tick for s in window] == list(range(1, 231))
    sub.close()

    with sqlite3.connect(tmp_path / "motion.db") as db:
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 230


def test_reset_keeps_tick_and_history(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([1.0, 1.0])
    sub.run_ticks(10)
    sub.reset()
    assert sub.state.position == [0.0, 0.0]
    assert sub.state.velocity == [0.0, 0.0]
    assert sub.state.tick == 10
    assert sub.store.count() == 10
    sub.reset(clear_trajectory=True)
    assert sub.store.count() == 0


def test_dims_mismatch_starts_at_rest(make_substrate) -> None:
    sub = make_substrate(dims=3)
    sub.set_target([1.0, 1.0, 1.0])
    sub.run_ticks(5)
    sub.close()
    again = make_substrate(dims=2)
    assert again.state.position == [0.0, 0.0]
    assert again.state.tick == 0
