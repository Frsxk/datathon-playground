import numpy as np

from scripts.reverse_engineer_v7_residuals import (
    boundary_distance,
    rank_percentile,
    target_midpoints,
)


def test_rank_percentile_is_monotone_and_bounded():
    values = np.array([3.0, 1.0, 2.0, 4.0])
    pct = rank_percentile(values)
    assert pct.tolist() == [0.625, 0.125, 0.375, 0.875]
    assert np.all((pct > 0) & (pct < 1))


def test_target_midpoints_encode_ordered_classes():
    assert target_midpoints(np.array([0, 1, 2, 3])).tolist() == [0.125, 0.375, 0.625, 0.875]


def test_boundary_distance_is_small_near_quartiles():
    score_pct = np.array([0.01, 0.249, 0.50, 0.751, 0.99])
    distance = boundary_distance(score_pct)
    assert distance[1] < distance[0]
    assert distance[2] == 0.0
    assert distance[3] < distance[4]
