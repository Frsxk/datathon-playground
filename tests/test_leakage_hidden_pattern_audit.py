import numpy as np
import pandas as pd

from scripts.audit_leakage_and_hidden_patterns import (
    build_interpretation,
    exact_feature_overlap,
    id_pattern_features,
    standardized_mean_difference,
)


def test_exact_feature_overlap_ignores_id_and_target():
    train = pd.DataFrame({
        "id": [1, 2],
        "a": [10, 20],
        "b": [0.5, 0.7],
        "target": [0, 1],
    })
    test = pd.DataFrame({
        "id": [99, 100],
        "a": [20, 30],
        "b": [0.7, 0.9],
    })
    result = exact_feature_overlap(train, test)
    assert result["overlap_count"] == 1
    assert result["train_duplicate_feature_rows"] == 0
    assert result["test_duplicate_feature_rows"] == 0


def test_standardized_mean_difference_is_zero_for_same_distribution():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert standardized_mean_difference(a, b) == 0.0


def test_id_pattern_features_include_mod_div_and_row_order():
    frame = pd.DataFrame({"id": [10, 11, 12, 13]})
    out = id_pattern_features(frame, modulus=3, divisor=5)
    assert out["id_mod_3"].tolist() == [1, 2, 0, 1]
    assert out["id_div_5"].tolist() == [2, 2, 2, 2]
    assert out["row_idx_mod_3"].tolist() == [0, 1, 2, 0]


def test_interpretation_does_not_treat_full_raw_as_structural_shortcut():
    report = {
        "exact_overlap": {"overlap_count": 0, "test_duplicate_feature_rows": 0},
        "feature_set_results": [
            {"feature_set": "full_raw", "hgb_accuracy": 0.48},
            {"feature_set": "id_patterns_m17", "hgb_accuracy": 0.27},
        ],
    }
    assert "No obvious row-order/ID/admin shortcut" in build_interpretation(report)
