# Submission Difference Audit

Generated at: `2026-07-09T12:24:00.423525+00:00`

## Inputs

- Old submission: `outputs/submission_production_v2.csv`
- New submission: `outputs/submission_production_v3.csv`

## Summary

- Rows: `800`
- IDs/order match: `true`
- Changed predictions: `128`
- Unchanged predictions: `672`
- Old target counts: `{0: 200, 1: 200, 2: 200, 3: 200}`
- New target counts: `{0: 200, 1: 200, 2: 200, 3: 200}`

## Transition matrix

Rows are old targets; columns are new targets.

|   old |   0 |   1 |   2 |   3 |
|------:|----:|----:|----:|----:|
|     0 | 180 |  20 |   0 |   0 |
|     1 |  20 | 162 |  18 |   0 |
|     2 |   0 |  18 | 156 |  26 |
|     3 |   0 |   0 |  26 | 174 |

## Delta counts

- Signed deltas: `{-1: 64, 1: 64}`
- Absolute deltas: `{1: 128}`

## First changed rows

|   id |   old_target |   new_target |   delta |
|-----:|-------------:|-------------:|--------:|
|   95 |            3 |            2 |      -1 |
|  104 |            1 |            0 |      -1 |
|  197 |            1 |            0 |      -1 |
|  205 |            0 |            1 |       1 |
|  208 |            2 |            3 |       1 |
|  255 |            2 |            3 |       1 |
|  263 |            1 |            0 |      -1 |
|  272 |            2 |            3 |       1 |
|  317 |            3 |            2 |      -1 |
|  326 |            2 |            1 |      -1 |
|  370 |            2 |            1 |      -1 |
|  384 |            2 |            1 |      -1 |
|  401 |            3 |            2 |      -1 |
|  416 |            0 |            1 |       1 |
|  489 |            0 |            1 |       1 |
|  525 |            2 |            3 |       1 |
|  557 |            2 |            3 |       1 |
|  588 |            1 |            0 |      -1 |
|  635 |            1 |            2 |       1 |
|  637 |            1 |            0 |      -1 |
|  669 |            2 |            3 |       1 |
|  687 |            2 |            1 |      -1 |
|  696 |            3 |            2 |      -1 |
|  717 |            0 |            1 |       1 |
|  747 |            0 |            1 |       1 |
|  760 |            1 |            0 |      -1 |
|  774 |            1 |            2 |       1 |
|  815 |            1 |            2 |       1 |
|  846 |            0 |            1 |       1 |
|  855 |            0 |            1 |       1 |
|  902 |            2 |            1 |      -1 |
|  928 |            2 |            3 |       1 |
|  970 |            0 |            1 |       1 |
|  982 |            2 |            1 |      -1 |
|  993 |            2 |            1 |      -1 |
| 1034 |            2 |            1 |      -1 |
| 1041 |            1 |            2 |       1 |
| 1054 |            1 |            2 |       1 |
| 1090 |            1 |            0 |      -1 |
| 1136 |            1 |            2 |       1 |

## Changed-vs-unchanged raw feature contrast

This is descriptive only; it uses no labels and trains no model.

| feature            |   changed_mean |   unchanged_mean |   mean_delta |   changed_median |   unchanged_median |
|:-------------------|---------------:|-----------------:|-------------:|-----------------:|-------------------:|
| kelas              |       414.117  |         400.924  |      13.1931 |          435     |            395.5   |
| tugas_diberikan    |        60.7578 |          63.5074 |      -2.7496 |           60.5   |             63.5   |
| tugas_selesai      |        29.7109 |          31.058  |      -1.3471 |           25.5   |             26.5   |
| skor_minat_belajar |        -0.2317 |          -0.0018 |      -0.2299 |           -0.26  |              0.07  |
| skor_kedisiplinan  |        -0.1134 |           0.0008 |      -0.1142 |           -0.11  |             -0.01  |
| skor_tryout        |        60.1234 |          60.0647 |       0.0587 |           61     |             60.1   |
| indeks_kehadiran   |         0.049  |           0.0445 |       0.0045 |            0.02  |              0.055 |
| skor_motivasi      |        -0.0314 |          -0.0298 |      -0.0016 |           -0.09  |             -0.015 |
| skor_literasi      |        -0.017  |          -0.0164 |      -0.0007 |           -0.205 |             -0.06  |

## Interpretation

- The public LB tied despite changed predictions, so the changed rows likely did not affect the public split enough to move the rounded score, or gains/losses canceled out.
- Use this audit to target boundary/threshold experiments rather than broad hyperparameter searches first.
- Because both submissions preserve a 200/200/200/200 class balance, the next likely lever is rank/threshold quality near class boundaries, not global class distribution.
