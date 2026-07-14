import numpy as np
import pandas as pd

from scripts.symbolic_hidden_rule_search import (
    build_symbolic_features,
    quartile_accuracy,
    rank_residual_correlation,
    zscore,
)


def test_zscore_handles_constant_vectors():
    out = zscore(np.array([5.0, 5.0, 5.0]))
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_quartile_accuracy_maps_scores_to_balanced_ordered_classes():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 10.0, 11.0, 12.0, 13.0])
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert quartile_accuracy(scores, y) == 1.0


def test_rank_residual_correlation_removes_baseline_linear_rank_signal():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    baseline = np.arange(len(y), dtype=float)
    feature = baseline * 10.0 + 5.0
    assert abs(rank_residual_correlation(feature, y, baseline)) < 1e-9


def test_build_symbolic_features_has_expected_target_free_columns():
    frame = pd.DataFrame({
        **{f"nilai_minggu_{i:02d}": np.arange(4) + i for i in range(1, 13)},
        **{f"aktivitas_hari_{i:02d}": np.arange(4) + i * 0.5 for i in range(1, 17)},
        "skor_motivasi": [1, 2, 3, 4],
        "skor_kedisiplinan": [4, 3, 2, 1],
        "tugas_selesai": [1, 2, 3, 4],
        "tugas_diberikan": [2, 2, 4, 4],
        "kelas": [1, 1, 2, 2],
        "urutan_ujian": [10, 20, 30, 40],
        "skor_tryout": [50, 60, 70, 80],
        "jarak_rumah_km": [1, 2, 3, 4],
        "skor_ekstrakurikuler": [1, 2, 3, 4],
        "indeks_kehadiran": [90, 80, 70, 60],
        "skor_literasi": [10, 20, 30, 40],
        "jumlah_saudara": [0, 1, 2, 3],
        "skor_minat_belajar": [2, 3, 4, 5],
        "id": [100, 101, 102, 103],
    })
    out = build_symbolic_features(frame)
    assert "target" not in out.columns
    for col in ["compl_x_wk_vol", "daily_period3_energy", "tryout_x_wk_slope", "motivation_discipline_balance"]:
        assert col in out.columns
    assert np.isfinite(out.to_numpy()).all()
