# Datathon Symbolic Hidden-Rule Search

Generated: `2026-07-14T10:44:10.352934+00:00`

## Safety

- Target-free formula screening only.
- Uses cached v7 OOF/test scores; no model retraining and no Kaggle submission.
- Cross-seed alpha transfer is diagnostic only and does not authorize a submission.

Baseline mean v7 OOF quartile accuracy: `0.627812`

## Top stable symbolic candidates

| feature | transfer gain | transfer accuracy | residual corr | train/test SMD | best in-sample alpha |
|---|---:|---:|---:|---:|---:|
| compl_x_wk_vol | +0.002500 | 0.624688 | -0.0154 | +0.0153 | +0.00 |
| exam_order_x_completion | +0.002062 | 0.624250 | -0.0006 | -0.0437 | +0.15 |
| compl_x_wk_slope | +0.000938 | 0.623125 | -0.0002 | +0.0568 | +0.08 |
| motivation_discipline_x_compl | +0.000750 | 0.622937 | +0.0263 | -0.0353 | +0.00 |
| literacy_x_tryout | +0.000625 | 0.622812 | -0.0096 | +0.0158 | +0.00 |
| tryout_x_wk_slope | +0.000563 | 0.622750 | -0.0012 | +0.0314 | +0.08 |
| kelas_x_tryout | +0.000563 | 0.622750 | +0.0211 | -0.0462 | +0.00 |
| weekly_late_minus_early | +0.000437 | 0.622625 | +0.0018 | +0.0281 | +0.08 |
| compl_x_daily_slope | +0.000000 | 0.622188 | -0.0062 | -0.0221 | +0.00 |
| compl_x_wk_range | -0.000062 | 0.622125 | -0.0160 | +0.0152 | +0.00 |
| daily_weekly_vol_ratio | -0.000187 | 0.622000 | +0.0093 | -0.0369 | +0.00 |
| tryout_x_daily_period3 | -0.000188 | 0.622000 | -0.0082 | -0.0057 | +0.00 |
| interest_x_attendance_x_compl | -0.000313 | 0.621875 | -0.0066 | -0.0056 | +0.00 |
| compl_x_shape_correlation | -0.000313 | 0.621875 | +0.0245 | -0.0124 | +0.00 |
| weekly_period3_energy | -0.000313 | 0.621875 | +0.0026 | +0.0509 | +0.00 |
| motivation_discipline_harmonic | -0.000437 | 0.621750 | -0.0149 | +0.0536 | +0.00 |
| daily_weekly_slope_product | -0.000500 | 0.621688 | +0.0257 | -0.0365 | +0.00 |
| compl_x_daily_vol | -0.000563 | 0.621625 | -0.0047 | -0.0476 | +0.08 |
| daily_period3_energy | -0.000750 | 0.621437 | -0.0129 | -0.0002 | +0.00 |
| daily_period5_energy | -0.000812 | 0.621375 | +0.0073 | -0.0129 | +0.00 |
| daily_late_minus_early | -0.001313 | 0.620875 | -0.0050 | -0.0491 | +0.00 |
| daily_weekly_shape_corr | -0.001375 | 0.620812 | +0.0249 | +0.0085 | +0.00 |
| kelas_x_weekly_vol | -0.001437 | 0.620750 | +0.0110 | +0.0020 | +0.00 |
| weekly_period2_energy | -0.001687 | 0.620500 | -0.0233 | -0.0027 | +0.00 |
| daily_period4_energy | -0.001687 | 0.620500 | -0.0173 | +0.0163 | -0.08 |

## Interpretation

At least one low-shift symbolic formula has a positive cross-seed diagnostic gain. It merits a separate, full nested-CV model-addition experiment before any Kaggle candidate is considered.
