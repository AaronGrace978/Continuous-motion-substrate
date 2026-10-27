from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronosoul_bridge import BaseDynamics, StaticAffectSource  # noqa: E402
from motion_substrate import ContinuousMotionSubstrate  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_substrate(tmp_path: Path, clock: FakeClock) -> Callable[..., ContinuousMotionSubstrate]:
    made: list[ContinuousMotionSubstrate] = []

    def factory(valence: float = 0.0, arousal: float = 0.4, jitter: bool = False,
                db: str = "motion.db", **kwargs: object) -> ContinuousMotionSubstrate:
        base = BaseDynamics() if jitter else replace(BaseDynamics(), jitter=0.0)
        kwargs.setdefault("catch_up", False)
        kwargs.setdefault("anticipate", False)
        kwargs.setdefault("target_half_life", math.inf)
        sub = ContinuousMotionSubstrate(
            str(tmp_path / db), affect_source=StaticAffectSource(valence, arousal),
            base=base, clock=clock, **kwargs,  # type: ignore[arg-type]
        )
        made.append(sub)
        return sub

    yield factory
    for sub in made:
        try:
            sub.store.close()
        except Exception:
            pass
