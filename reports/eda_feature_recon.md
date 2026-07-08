# Datathon 2026 Playground — EDA & Feature Recon

## Dataset Snapshot

- Train shape: `(3200, 43)`
- Test shape: `(800, 42)`
- Missing values train: `0`
- Missing values test: `0`
- Target counts: `{0: 813, 1: 796, 2: 784, 3: 807}`
- Classes are balanced enough that accuracy is a reasonable metric.

## Baseline Results

- **xgboost_fe**: 0.49031 ± 0.01378 folds=[0.4828125, 0.5140625, 0.4953125, 0.4734375, 0.4859375]
- **histgb_fe**: 0.48844 ± 0.01364 folds=[0.484375, 0.4875, 0.5140625, 0.4734375, 0.4828125]
- **logreg_fe_scaled**: 0.47469 ± 0.02033 folds=[0.5078125, 0.4796875, 0.475, 0.4453125, 0.465625]
- **randomforest_fe**: 0.46281 ± 0.00913 folds=[0.4703125, 0.459375, 0.4765625, 0.4546875, 0.453125]
- **extratrees_fe**: 0.45687 ± 0.01103 folds=[0.45625, 0.4703125, 0.4640625, 0.4375, 0.45625]
- **logreg_raw_scaled**: 0.33719 ± 0.01847 folds=[0.3515625, 0.3328125, 0.303125, 0.353125, 0.3453125]

Best local CV model in this first loop: **xgboost_fe** (0.49031).

## Top Raw Feature Correlations with Target

|                    |      0 |
|:-------------------|-------:|
| tugas_selesai      | 0.215  |
| skor_tryout        | 0.1142 |
| aktivitas_hari_12  | 0.0447 |
| aktivitas_hari_14  | 0.0325 |
| tugas_diberikan    | 0.0324 |
| aktivitas_hari_06  | 0.0314 |
| aktivitas_hari_08  | 0.0271 |
| aktivitas_hari_11  | 0.027  |
| skor_minat_belajar | 0.0266 |
| indeks_kehadiran   | 0.0225 |
| jarak_rumah_km     | 0.0196 |
| aktivitas_hari_07  | 0.0194 |
| aktivitas_hari_10  | 0.0192 |
| aktivitas_hari_03  | 0.0166 |
| jumlah_saudara     | 0.016  |

## Top Engineered Feature Correlations with Target

|                        |      0 |
|:-----------------------|-------:|
| tugas_completion_gap   | 0.4045 |
| tugas_completion_ratio | 0.4045 |
| minggu_std             | 0.3956 |
| minggu_iqr             | 0.3838 |
| minggu_range           | 0.3746 |
| minggu_q75             | 0.3537 |
| minggu_q25             | 0.344  |
| minggu_max             | 0.3384 |
| minggu_min             | 0.3253 |
| tugas_remaining        | 0.2639 |
| tugas_selesai          | 0.215  |
| minggu_diff_std        | 0.2069 |
| hari_diff_abs_mean     | 0.1869 |
| hari_diff_std          | 0.1817 |
| minggu_diff_abs_mean   | 0.1618 |
| skor_tryout            | 0.1142 |
| hari_range             | 0.0503 |
| aktivitas_hari_12      | 0.0447 |
| hari_max               | 0.0442 |
| hari_std               | 0.0393 |

## Recommended Next Feature Pounces

1. **Sequence trend features**: keep weekly/day slopes, late-vs-early deltas, diff volatility, positive/negative step counts.
2. **Assignment behavior**: `tugas_selesai / tugas_diberikan`, remaining tasks, and interactions with discipline/motivation.
3. **Activity consistency**: day std/IQR/diff_abs_mean may capture stable vs erratic learning patterns.
4. **Academic trajectory**: interactions between `skor_tryout`, weekly slope, literacy, and attendance.
5. **Class identifier caution**: `kelas` may encode groups. Frequency/mod features are target-safe; target encoding must be done inside CV folds only to avoid leakage.
6. **Ordered labels**: metric is accuracy, but labels have order. Try regression/ordinal-inspired features or calibrated class boundaries later, while still submitting class labels.
7. **Model stack**: compare ExtraTrees/RandomForest/HistGradientBoosting/XGBoost, then blend if CV is stable.