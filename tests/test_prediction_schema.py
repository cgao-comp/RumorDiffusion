from __future__ import annotations

import pytest

from Libs.utils.artifact_schema import validate_prediction_row


def _row():
    return {
        "run_id": "run-1", "dataset": "twitter15", "split": "test", "source_id": "s1",
        "horizon_unit": "minutes", "horizon_value": 10, "model": "rumordiffusion",
        "model_variant": "canonical", "y_true": 1, "logits": [0.1, 0.9],
        "probs_uncalibrated": [0.31, 0.69], "probs_calibrated": [0.3, 0.7],
        "y_pred": 1, "confidence": 0.7, "uncertainty_score": 0.02,
        "accepted": True, "decision": "accept", "split_seed": 42, "ctdm_seed": 42,
        "arbiter_seed": 42, "inference_seed": 42, "calibration_tag": "val_temperature", "sampling_S": 1,
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64, "split_sha256": "c" * 64,
    }


def test_prediction_row_validates():
    validate_prediction_row(_row())


@pytest.mark.parametrize("field", ["source_id", "logits", "probs_uncalibrated"])
def test_prediction_row_missing_required_field_fails(field):
    row = _row()
    row.pop(field)
    with pytest.raises(ValueError):
        validate_prediction_row(row)


def test_prediction_row_rejects_nan():
    row = _row()
    row["confidence"] = float("nan")
    with pytest.raises(ValueError):
        validate_prediction_row(row)
