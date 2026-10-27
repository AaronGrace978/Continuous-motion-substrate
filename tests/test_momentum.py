from __future__ import annotations

import pytest

from chronosoul_bridge import SpringParams
from momentum import Anticipator, InputVelocityEstimator, kinematic_extrapolate, predict_state


def test_one_tick_prediction_is_exact_without_noise(make_substrate) -> None:
    sub = make_substrate()
    sub.set_target([1.0, -1.0])
    sub.tick_once()
    for _ in range(20):
        predicted = sub.predict(sub.dt)["position"]
        assert sub.tick_once().position == pytest.approx(predicted, abs=1e-12)


def test_input_velocity_estimator_converges_on_ramp() -> None:
    est = InputVelocityEstimator(dims=1, smoothing_time=0.3)
    for i in range(50):
        est.observe([0.5 * i * 0.1], 0.1)
    assert est.velocity == pytest.approx([0.5], rel=1e-3)


def test_lead_target_leads_by_lag_time() -> None:
    params = SpringParams(stiffness=12.0, mass=1.0, damping_ratio=0.9, momentum_gain=1.0, jitter=0.0)
    ant = Anticipator(dims=1)
    for i in range(50):
        ant.observe_input([i * 0.1], 0.1)
    lead = ant.lead_target([5.0], params)[0] - 5.0
    assert lead == pytest.approx(params.damping / params.stiffness * 1.0, rel=1e-3)
    ant.enabled = False
    assert ant.lead_target([5.0], params) == [5.0]


def test_anticipation_removes_ramp_lag(make_substrate) -> None:
    def mean_lag(anticipate: bool, db: str) -> float:
        sub = make_substrate(anticipate=anticipate, db=db, dims=1)
        lags = []
        for i in range(100):
            target = 0.5 * (i + 1) * sub.dt
            sub.set_target([target])
            x = sub.tick_once(record=False).position[0]
            if i >= 50:
                lags.append(target - x)
        return sum(lags) / len(lags)

    reactive, anticipating = mean_lag(False, "r.db"), mean_lag(True, "a.db")
    assert reactive > 0.2
    assert abs(anticipating) < 0.25 * reactive


def test_kinematic_extrapolation_agrees_short_term() -> None:
    params = SpringParams(stiffness=12.0, mass=1.0, damping_ratio=0.9, momentum_gain=1.0, jitter=0.0)
    rollout = predict_state([0.2], [1.0], [1.0], params, horizon=0.1, dt=0.1, substeps=16)
    taylor = kinematic_extrapolate([0.2], [1.0], [1.0], params, horizon=0.1)
    assert taylor.position[0] == pytest.approx(rollout.position[0], abs=0.01)
