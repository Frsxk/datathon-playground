import numpy as np
import pandas as pd

from scripts.exp_v10_exam_order_completion import add_exam_order_completion_feature


def test_exam_order_completion_feature_uses_completed_over_assigned_tasks():
    frame = pd.DataFrame({
        "urutan_ujian": [10, 20, 30],
        "tugas_selesai": [1, 4, 3],
        "tugas_diberikan": [2, 4, 0],
    })
    out = add_exam_order_completion_feature(frame)
    assert out.tolist() == [5.0, 20.0, 90.0]
    assert np.isfinite(out).all()
