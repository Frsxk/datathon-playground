import numpy as np
import pandas as pd

from scripts.advanced_generator_fingerprint_audit import (
    cluster_label_purity,
    decimal_places,
    monotone_quartile_accuracy,
)


def test_decimal_places_detects_simple_grid():
    assert decimal_places(pd.Series([1.0, 1.5, 2.25])) == 2
    assert decimal_places(pd.Series([1, 2, 3])) == 0


def test_cluster_label_purity_weighted_by_cluster_size():
    clusters = np.array([0, 0, 0, 1, 1])
    y = np.array([1, 1, 2, 3, 3])
    out = cluster_label_purity(clusters, y)
    assert out["weighted_purity"] == 0.8
    assert out["n_clusters"] == 2


def test_monotone_quartile_accuracy_detects_ordered_signal():
    values = np.array([0.1, 0.2, 0.3, 0.4, 10.0, 11.0, 12.0, 13.0])
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert monotone_quartile_accuracy(values, y) == 1.0
