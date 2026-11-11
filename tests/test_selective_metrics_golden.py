import numpy as np
import pytest

from Libs.utils.selective_metrics import (
    aurc,
    coverage,
    coverage_at_target_risk,
    deployment_cost,
    e_aurc,
    max_probability_confidence,
    predictive_entropy,
    resolved_accuracy,
    resolved_macro_f1,
    risk_at_coverage,
    risk_coverage_curve,
    select_reject_threshold,
    selective_risk,
    trajectory_variance,
)


Y_TRUE = np.array([0, 1, 1, 0])
Y_PRED = np.array([0, 0, 1, 1])
CONFIDENCE = np.array([0.95, 0.90, 0.80, 0.70])


def test_coverage_and_resolved_metrics():
    accepted = np.array([True, True, False, False])
    assert coverage(accepted) == pytest.approx(0.5)
    assert selective_risk(Y_TRUE, Y_PRED, accepted) == pytest.approx(0.5)
    assert resolved_accuracy(Y_TRUE, Y_PRED, accepted) == pytest.approx(0.5)
    assert resolved_macro_f1(Y_TRUE, Y_PRED, accepted, labels=[0, 1]) == pytest.approx(1 / 3)


def test_risk_coverage_curve_and_aurc():
    curve = risk_coverage_curve(Y_TRUE, Y_PRED, CONFIDENCE)
    np.testing.assert_allclose(curve["coverage"], [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(curve["risk"], [0, 0, 0.5, 1 / 3, 0.5])
    assert aurc(Y_TRUE, Y_PRED, CONFIDENCE) == pytest.approx(0.2708333333)
    assert e_aurc(Y_TRUE, Y_PRED, CONFIDENCE) >= 0


def test_risk_and_coverage_targets():
    assert risk_at_coverage(Y_TRUE, Y_PRED, CONFIDENCE, 0.5) == pytest.approx(0.5)
    assert coverage_at_target_risk(Y_TRUE, Y_PRED, CONFIDENCE, 0.0) == pytest.approx(0.25)


def test_reject_threshold_is_selected_on_validation_only():
    probs = np.array([[0.95, 0.05], [0.90, 0.10], [0.20, 0.80], [0.30, 0.70]])
    threshold = select_reject_threshold(Y_TRUE, probs, target_risk=0.35, split="val")
    assert threshold == pytest.approx(0.8)
    with pytest.raises(ValueError, match="validation-only"):
        select_reject_threshold(Y_TRUE, probs, target_risk=0.35, split="test")


def test_confidence_interfaces():
    probs = np.array([[0.75, 0.25], [0.5, 0.5]])
    np.testing.assert_allclose(max_probability_confidence(probs), [0.75, 0.5])
    np.testing.assert_allclose(predictive_entropy(probs), [-0.75 * np.log(0.75) - 0.25 * np.log(0.25), np.log(2)])
    np.testing.assert_allclose(predictive_entropy(probs, base=2), [0.8112781245, 1.0])


def test_trajectory_variance_and_deployment_cost():
    trajectories = np.array(
        [
            [[0.8, 0.2], [0.7, 0.3]],
            [[0.6, 0.4], [0.7, 0.3]],
        ]
    )
    np.testing.assert_allclose(trajectory_variance(trajectories), [0.0025, 0.0025])
    accepted = np.array([True, False, True, False])
    breakdown = deployment_cost(
        Y_TRUE,
        Y_PRED,
        accepted,
        fp_cost=2,
        fn_cost=3,
        reject_cost=1,
        delay_cost=0.5,
        delay=[1, 2, 3, 4],
        return_breakdown=True,
    )
    assert breakdown == {
        "false_positive": 0.0,
        "false_negative": 0.0,
        "reject": 2.0,
        "delay": 5.0,
        "total": 7.0,
    }


@pytest.mark.parametrize(
    "func,args",
    [
        (selective_risk, ([], [], [])),
        (risk_coverage_curve, ([0], [0], [np.nan])),
        (predictive_entropy, ([[1.0, -0.1]],)),
        (trajectory_variance, ([[1.0]],)),
    ],
)
def test_invalid_or_empty_inputs_fail_closed(func, args):
    with pytest.raises(ValueError):
        func(*args)


def test_shape_mismatch_fails_closed():
    with pytest.raises(ValueError):
        selective_risk([0, 1], [0], [True, True])
    with pytest.raises(ValueError):
        risk_coverage_curve([0], [0], [0.9, 0.8])
