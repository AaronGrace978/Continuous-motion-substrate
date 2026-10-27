"""Measure what the emotional coupling and the anticipation layer actually do.

    python motion_eval.py            # table
    python motion_eval.py --json     # machine-readable

Everything is simulated, deterministic (seeded), and offline. The README's
numbers come from this script; the tests assert the same orderings.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from typing import Any

from chronosoul_bridge import BaseDynamics, StaticAffectSource
from motion_substrate import ContinuousMotionSubstrate

MOODS: dict[str, tuple[float, float]] = {
    "calm": (0.6, 0.0),       # pleasant, quiescent
    "neutral": (0.0, 0.4),    # a' = 0, v = 0: the base spring
    "anxious": (-0.8, 0.95),  # unpleasant, highly aroused
}


def make(mood: str, jitter: bool = True, anticipate: bool = False, seed: int = 7) -> ContinuousMotionSubstrate:
    valence, arousal = MOODS[mood]
    base = BaseDynamics() if jitter else replace(BaseDynamics(), jitter=0.0)
    return ContinuousMotionSubstrate(
        ":memory:", dims=1, affect_source=StaticAffectSource(valence, arousal),
        base=base, seed=seed, anticipate=anticipate, catch_up=False, target_half_life=math.inf,
    )


def step_response(mood: str, seconds: float = 8.0) -> dict[str, float]:
    """Unit step from rest, noise off: overshoot, 10-90% rise time, 2% settling time."""
    sub = make(mood, jitter=False)
    sub.set_target([1.0])
    xs: list[float] = []
    for _ in range(int(seconds * sub.tick_hz)):
        xs.append(sub.tick_once(record=False).position[0])
    dt = sub.dt
    t10 = next(i for i, x in enumerate(xs) if x >= 0.1) * dt
    t90 = next(i for i, x in enumerate(xs) if x >= 0.9) * dt
    last_out = max((i for i, x in enumerate(xs) if abs(x - 1.0) > 0.02), default=-1)
    return {
        "overshoot_pct": max(0.0, max(xs) - 1.0) * 100.0,
        "rise_time_s": t90 - t10 + dt,
        "settle_time_s": (last_out + 1) * dt,
    }


def jitter_rms(mood: str, seconds: float = 30.0) -> float:
    """RMS distance from a held target once settled, noise on."""
    sub = make(mood, jitter=True)
    for _ in range(int(5 * sub.tick_hz)):
        sub.tick_once(record=False)
    sq = 0.0
    n = int(seconds * sub.tick_hz)
    for _ in range(n):
        sq += sub.tick_once(record=False).position[0] ** 2
    return math.sqrt(sq / n)


def ramp_lag(mood: str, anticipate: bool, speed: float = 0.5, seconds: float = 10.0) -> float:
    """Mean (target - position) over the second half of a constant-speed ramp, noise off.

    Positive: the state trails the input. Negative: it runs ahead of it.
    """
    sub = make(mood, jitter=False, anticipate=anticipate)
    n = int(seconds * sub.tick_hz)
    errs: list[float] = []
    for i in range(n):
        target = speed * (i + 1) * sub.dt
        sub.set_target([target])
        x = sub.tick_once(record=False).position[0]
        if i >= n // 2:
            errs.append(target - x)
    return sum(errs) / len(errs)


def prediction_error(mood: str, horizon: float = 0.5, seconds: float = 30.0) -> dict[str, float]:
    """Error of the substrate's own forecast vs. what then happened.

    one_step_noise_free: one tick ahead, held target, jitter off (should be ~0).
    horizon_rms: `horizon` seconds ahead, jitter on, while the input keeps moving
    (target = sin(0.8 t)). The forecast assumes the led target is held, so this
    is the honest error of acting ahead of an input that is still changing.
    """
    sub = make(mood, jitter=False)
    sub.set_target([1.0])
    sub.tick_once(record=False)
    worst = 0.0
    for _ in range(50):
        predicted = sub.predict(sub.dt)["position"][0]
        actual = sub.tick_once(record=False).position[0]
        worst = max(worst, abs(predicted - actual))

    sub = make(mood, jitter=True, anticipate=True)
    steps = int(round(horizon / sub.dt))
    n = int(seconds * sub.tick_hz)
    positions: list[float] = []
    forecasts: list[float] = []
    for i in range(n + steps):
        sub.set_target([math.sin(0.8 * (i + 1) * sub.dt)])
        positions.append(sub.tick_once(record=False).position[0])
        forecasts.append(sub.predict(horizon)["position"][0])
    warmup = int(5 * sub.tick_hz)
    errs = [forecasts[i] - positions[i + steps] for i in range(warmup, n)]
    return {"one_step_noise_free": worst, "horizon_rms": math.sqrt(sum(e * e for e in errs) / len(errs))}


def tick_throughput(seconds: float = 1.0) -> float:
    """Recorded ticks per second, single thread, in-memory SQLite. Machine-dependent."""
    sub = ContinuousMotionSubstrate(":memory:", dims=2, affect_source=StaticAffectSource(), catch_up=False)
    n = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        sub.tick_once()
        n += 1
    return n / (time.perf_counter() - start)


def run() -> dict[str, Any]:
    out: dict[str, Any] = {"moods": {}}
    for mood in MOODS:
        sub = make(mood)
        params = sub.tick_once(record=False).params
        out["moods"][mood] = {
            "valence": MOODS[mood][0], "arousal": MOODS[mood][1],
            "params": {k: round(v, 3) for k, v in params.to_dict().items()},
            "step": step_response(mood),
            "jitter_rms": jitter_rms(mood),
            "ramp_lag_reactive": ramp_lag(mood, anticipate=False),
            "ramp_lag_anticipating": ramp_lag(mood, anticipate=True),
            "prediction": prediction_error(mood),
        }
    out["ticks_per_second"] = tick_throughput()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("| mood | k | m | zeta | overshoot | rise 10-90 | settle 2% | jitter RMS | ramp lag reactive | ramp lag anticipating | 0.5 s forecast RMS |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mood, r in result["moods"].items():
        p, s = r["params"], r["step"]
        print(f"| {mood} | {p['stiffness']:.2f} | {p['mass']:.2f} | {p['damping_ratio']:.2f} "
              f"| {s['overshoot_pct']:.1f}% | {s['rise_time_s']:.1f} s | {s['settle_time_s']:.1f} s "
              f"| {r['jitter_rms']:.4f} | {r['ramp_lag_reactive']:+.3f} | {r['ramp_lag_anticipating']:+.3f} "
              f"| {r['prediction']['horizon_rms']:.4f} |")
    worst = max(r["prediction"]["one_step_noise_free"] for r in result["moods"].values())
    print(f"\nOne-tick noise-free forecast, worst error over all moods: {worst:.2e}")
    print(f"Recorded ticks per second (this machine, in-memory SQLite): {result['ticks_per_second']:.0f}")


if __name__ == "__main__":
    main()
