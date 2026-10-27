"""Trajectory memory: the path the state took, not only where it ended.

- TrajectoryStore: SQLite store with ChronoSoul's two-tier layout. The newest
  `hot_limit` samples are tier 'hot'; older samples are moved to tier 'cold'
  and never deleted. Reads span both tiers transparently.
- interpolate(): cubic Hermite between two ticks, using the stored velocities,
  so the curve between samples has the right slope and not just the right ends.
- replay(): resample a window of the trajectory at any rate.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

Vector = list[float]

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
    tick INTEGER PRIMARY KEY, sim_time REAL NOT NULL, wall_time REAL NOT NULL,
    position TEXT NOT NULL, velocity TEXT NOT NULL, target TEXT NOT NULL,
    valence REAL, arousal REAL, tier TEXT NOT NULL DEFAULT 'hot');
CREATE INDEX IF NOT EXISTS idx_samples_tier ON samples(tier, tick);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class Sample:
    tick: int
    sim_time: float
    wall_time: float
    position: Vector
    velocity: Vector
    target: Vector
    valence: float = 0.0
    arousal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick, "sim_time": self.sim_time, "wall_time": self.wall_time,
            "position": self.position, "velocity": self.velocity, "target": self.target,
            "valence": self.valence, "arousal": self.arousal,
        }


@dataclass(frozen=True)
class Frame:
    """One interpolated point of a replay."""

    sim_time: float
    position: Vector
    velocity: Vector

    def to_dict(self) -> dict[str, Any]:
        return {"sim_time": self.sim_time, "position": self.position, "velocity": self.velocity}


class TrajectoryStore:
    def __init__(self, path: str, hot_limit: int = 6000, archive_every: int = 100) -> None:
        self.path = path
        self.hot_limit = hot_limit
        self.archive_every = archive_every
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._since_archive = 0
        with self._lock:
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()

    def commit(self) -> None:
        with self._lock:
            self._db.commit()

    # ---- samples
    def append(self, sample: Sample) -> None:
        """Insert one sample. Commits are batched with archiving; call `commit()` to flush."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO samples(tick, sim_time, wall_time, position, velocity, target, "
                "valence, arousal, tier) VALUES (?,?,?,?,?,?,?,?, 'hot')",
                (sample.tick, sample.sim_time, sample.wall_time, json.dumps(sample.position),
                 json.dumps(sample.velocity), json.dumps(sample.target), sample.valence, sample.arousal),
            )
            self._since_archive += 1
            if self._since_archive >= self.archive_every:
                self._since_archive = 0
                self.archive_overflow()
                self._db.commit()

    def archive_overflow(self) -> int:
        """Move samples older than the hot window to the cold tier. Returns how many moved."""
        with self._lock:
            row = self._db.execute("SELECT MAX(tick) FROM samples").fetchone()
            if row[0] is None:
                return 0
            cutoff = int(row[0]) - self.hot_limit
            cur = self._db.execute("UPDATE samples SET tier='cold' WHERE tier='hot' AND tick <= ?", (cutoff,))
            return cur.rowcount

    def count(self, tier: Optional[str] = None) -> int:
        with self._lock:
            if tier is None:
                return self._db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM samples WHERE tier=?", (tier,)).fetchone()[0]

    def window(self, start_tick: int, end_tick: int) -> list[Sample]:
        """Samples with start_tick <= tick <= end_tick from both tiers, oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM samples WHERE tick BETWEEN ? AND ? ORDER BY tick", (start_tick, end_tick)
            ).fetchall()
        return [_row_to_sample(r) for r in rows]

    def latest(self, n: int) -> list[Sample]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM samples ORDER BY tick DESC LIMIT ?", (n,)).fetchall()
        return [_row_to_sample(r) for r in reversed(rows)]

    def last_tick(self) -> Optional[int]:
        with self._lock:
            row = self._db.execute("SELECT MAX(tick) FROM samples").fetchone()
        return None if row[0] is None else int(row[0])

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM samples")
            self._db.commit()

    # ---- meta (substrate state lives here, like ChronoSoul's store metadata)
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
            self._db.commit()


def _row_to_sample(row: sqlite3.Row) -> Sample:
    return Sample(
        tick=int(row["tick"]), sim_time=float(row["sim_time"]), wall_time=float(row["wall_time"]),
        position=json.loads(row["position"]), velocity=json.loads(row["velocity"]),
        target=json.loads(row["target"]), valence=float(row["valence"] or 0.0),
        arousal=float(row["arousal"] or 0.0),
    )


# --------------------------------------------------------------------------- interpolation

def hermite(a: Sample, b: Sample, sim_time: float) -> Frame:
    """Cubic Hermite interpolation between two samples using their positions and velocities."""
    h = b.sim_time - a.sim_time
    if h <= 0:
        return Frame(a.sim_time, list(a.position), list(a.velocity))
    s = max(0.0, min(1.0, (sim_time - a.sim_time) / h))
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    d00, d10, d01, d11 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1, -6 * s2 + 6 * s, 3 * s2 - 2 * s
    position = [h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
                for p0, v0, p1, v1 in zip(a.position, a.velocity, b.position, b.velocity)]
    velocity = [(d00 * p0 + d10 * h * v0 + d01 * p1 + d11 * h * v1) / h
                for p0, v0, p1, v1 in zip(a.position, a.velocity, b.position, b.velocity)]
    return Frame(a.sim_time + s * h, position, velocity)


def interpolate(samples: Sequence[Sample], sim_time: float) -> Frame:
    """State at an arbitrary sim time, clamped to the ends of `samples` (which must be sorted)."""
    if not samples:
        raise ValueError("cannot interpolate an empty trajectory")
    times = [s.sim_time for s in samples]
    if sim_time <= times[0]:
        first = samples[0]
        return Frame(first.sim_time, list(first.position), list(first.velocity))
    if sim_time >= times[-1]:
        last = samples[-1]
        return Frame(last.sim_time, list(last.position), list(last.velocity))
    i = bisect.bisect_right(times, sim_time)
    return hermite(samples[i - 1], samples[i], sim_time)


def replay(samples: Sequence[Sample], rate_hz: float,
           start: Optional[float] = None, end: Optional[float] = None) -> Iterator[Frame]:
    """Resample a trajectory window at `rate_hz` (which may be above or below the tick rate)."""
    if not samples:
        return
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    t0 = samples[0].sim_time if start is None else start
    t1 = samples[-1].sim_time if end is None else end
    step = 1.0 / rate_hz
    n = int(round((t1 - t0) / step + 1e-9))
    times = [s.sim_time for s in samples]
    for k in range(n + 1):
        t = t0 + k * step
        if t <= times[0] or t >= times[-1]:
            yield interpolate(samples, t)
            continue
        i = bisect.bisect_right(times, t)
        yield hermite(samples[i - 1], samples[i], t)


def path_length(samples: Sequence[Sample]) -> float:
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        total += sum((p1 - p0) ** 2 for p0, p1 in zip(a.position, b.position)) ** 0.5
    return total
