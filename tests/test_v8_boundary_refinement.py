import numpy as np

from scripts.exp_v8_boundary_refinement import (
    apply_thresholds,
    balanced_quartile_thresholds,
    candidate_offsets,
    load_reference_predictions,
    passes_balance_guard,
)


def test_balanced_thresholds_produce_four_equal_bins():
    scores = np.arange(40, dtype=float)
    cuts = balanced_quartile_thresholds(scores)
    pred = apply_thresholds(scores, cuts)
    assert cuts.shape == (3,)
    assert np.bincount(pred, minlength=4).tolist() == [10, 10, 10, 10]


def test_threshold_application_is_monotone_and_clipped():
    scores = np.array([-10.0, 0.0, 1.0, 2.0, 99.0])
    pred = apply_thresholds(scores, np.array([0.0, 1.0, 2.0]))
    assert pred.tolist() == [0, 0, 1, 2, 3]


def test_candidate_offsets_are_deterministic_and_include_baseline():
    offsets = candidate_offsets((-8, 0, 8), step=8)
    assert offsets[0] == (0, 0, 0)
    assert len(offsets) == 27
    assert offsets == candidate_offsets((-8, 0, 8), step=8)


def test_balance_guard_accepts_near_quartile_and_rejects_large_drift():
    assert passes_balance_guard(np.array([200, 184, 216, 200]), target_per_class=200, max_drift=16)
    assert not passes_balance_guard(np.array([201, 184, 152, 263]), target_per_class=200, max_drift=16)


def test_reference_predictions_follow_submitted_id_order(tmp_path):
    path = tmp_path / "submission_v7.csv"
    path.write_text("id,target\n11,3\n22,1\n")
    result = load_reference_predictions(path, pd_ids=np.array([11, 22]))
    assert result.tolist() == [3, 1]
