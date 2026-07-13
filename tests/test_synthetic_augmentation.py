import numpy as np
import pandas as pd

from scripts.exp_synthetic_augmentation import (
    DAILY_COLS,
    WEEKLY_COLS,
    augment_training_frame,
    synthesize_class_rows,
)


def tiny_frame():
    rows = []
    for target in range(4):
        for i in range(2):
            row = {"id": target * 10 + i, "target": target,
                   "tugas_selesai": 4 + target + i, "tugas_diberikan": 8 + i,
                   "kelas": 100 + target, "urutan_ujian": 0.2 * i,
                   "skor_tryout": float(target), "skor_motivasi": float(target),
                   "skor_kedisiplinan": float(target + 1), "skor_ekstrakurikuler": 0.0,
                   "indeks_kehadiran": 0.0, "skor_literasi": 0.0,
                   "jumlah_saudara": 0.0, "skor_minat_belajar": 0.0,
                   "jarak_rumah_km": 0.0}
            row.update({c: float(target + j + i) for j, c in enumerate(WEEKLY_COLS)})
            row.update({c: float(target + j + i) for j, c in enumerate(DAILY_COLS)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_synthetic_rows_preserve_labels_and_row_count():
    frame = tiny_frame()
    out = synthesize_class_rows(frame, n_rows=12, rng=np.random.default_rng(7), id_start=-1)
    assert len(out) == 12
    assert set(out["target"]).issubset(set(frame["target"]))
    assert set(out["id"]).isdisjoint(set(frame["id"]))


def test_synthetic_rows_preserve_task_constraint_and_sequence_columns():
    frame = tiny_frame()
    out = synthesize_class_rows(frame, n_rows=12, rng=np.random.default_rng(8), id_start=-1)
    assert (out["tugas_selesai"] <= out["tugas_diberikan"]).all()
    assert list(out.columns) == list(frame.columns)
    assert out[WEEKLY_COLS + DAILY_COLS].notna().all().all()


def test_zero_factor_returns_an_exact_copy():
    frame = tiny_frame()
    out = augment_training_frame(frame, factor=0.0, rng=np.random.default_rng(9))
    pd.testing.assert_frame_equal(out, frame)
