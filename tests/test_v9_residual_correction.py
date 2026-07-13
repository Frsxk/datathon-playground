import numpy as np

from scripts.exp_v9_residual_correction import (
    apply_residual_correction,
    residual_targets,
    select_best_config,
)


def test_residual_targets_are_target_midpoint_minus_score_percentile():
    y = np.array([0, 1, 2, 3])
    score_pct = np.array([0.10, 0.40, 0.60, 0.90])
    got = residual_targets(y, score_pct)
    assert np.allclose(got, np.array([0.025, -0.025, 0.025, -0.025]))


def test_apply_residual_correction_clips_to_unit_interval():
    base = np.array([0.05, 0.50, 0.95])
    corr = np.array([-1.0, 0.2, 1.0])
    got = apply_residual_correction(base, corr, alpha=0.5)
    assert np.allclose(got, np.array([0.0, 0.60, 1.0]))


def test_select_best_config_prefers_improvement_then_simplicity():
    rows = [
        {"method": "hgb", "alpha": 0.3, "mean_accuracy": 0.62, "std_accuracy": 0.02},
        {"method": "ridge", "alpha": 0.1, "mean_accuracy": 0.62, "std_accuracy": 0.01},
        {"method": "ridge", "alpha": 0.0, "mean_accuracy": 0.61, "std_accuracy": 0.0},
    ]
    best = select_best_config(rows)
    assert best["method"] == "ridge"
    assert best["alpha"] == 0.1
