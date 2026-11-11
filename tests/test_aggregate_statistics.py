from __future__ import annotations

import pytest

from Libs.utils.metrics import holm_bonferroni, paired_bootstrap, seed_level_summary


def test_seed_summary_uses_sample_sd_and_student_t_ci():
    result = seed_level_summary([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["n"] == 5
    assert result["df"] == 4
    assert result["mean"] == pytest.approx(0.3)
    assert result["sample_sd"] > 0.0
    assert result["ci95"][0] < result["mean"] < result["ci95"][1]


def test_paired_bootstrap_requires_identical_ids():
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap({"a": 1.0}, {"b": 0.0})


def test_holm_correction_is_monotonic_in_sorted_order():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
