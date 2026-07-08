# Datathon 2026 Playground — Competition Brief

Generated: 2026-07-08T06:48:27.930115+00:00

## Status

- Playground window from broadcast: **July 8–17, 2026**.
- Broadcast states this playground is **learning-only** and **not part of competition evaluation**.
- Kaggle slug: `datathon-playground-2026`.
- Kaggle CLI access verified via `~/.kaggle/access_token`.

## Objective

Schools often seek to identify students who may benefit from additional support early in the semester. Using academic and behavioral records, your task is to predict each student's performance level among four categories.

This competition focuses on multi-class classification using tabular data containing information related to grades, learning activity, assignments, and several additional attributes.

## Labels

Target `target` is a 4-class ordered performance label:

- `0` — Very Low Performance
- `1` — Low Performance
- `2` — Good Performance
- `3` — Excellent Performance

## Evaluation

Submissions are evaluated using **multi-class classification accuracy**, defined as the proportion of correctly predicted labels over all samples in the test set.

### Submission File

For every `id` in the test set, predict the corresponding `target` value:

```csv
id,target
0,2
3,0
5,1
...
```

The submission file must contain a header and follow the format shown above.

## Files and Shapes

- `train.csv`: 3200 rows × 43 columns
- `test.csv`: 800 rows × 42 columns
- `sample_submission.csv`: 800 rows × 2 columns

## Submission Format

Must contain exactly:

```csv
id,target
```

One row per test `id`; `target` must be one of 0, 1, 2, 3.

## Data Description Extract

The competition provides the following files:

* **`train.csv`** — training data containing input features and the `target` column.
* **`test.csv`** — test data without target labels.
* **`sample_submission.csv`** — an example submission file in the correct format.

### Features

| Group | Columns                                                                                                               | Description                                                              |
| ----- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **A** | `nilai_minggu_01` … `nilai_minggu_12`                                                                                 | Weekly grade changes across a 12-week semester.                          |
| **B** | `skor_motivasi`, `skor_kedisiplinan`                                                                                  | Motivation and discipline assessment scores.                             |
| **C** | `aktivitas_hari_01` … `aktivitas_hari_16`                                                                             | Daily learning activity indices recorded over 16 consecutive days.       |
| **D** | `tugas_selesai`, `tugas_diberikan`                                                                                    | Numbers of completed and assigned tasks.                                 |
| **E** | `kelas`, `urutan_ujian`, `skor_tryout`                                                                                | Class identifier, exam order, and mock exam score.                       |
| —     | `jarak_rumah_km`, `skor_ekstrakurikuler`, `indeks_kehadiran`, `skor_literasi`, `jumlah_saudara`, `skor_minat_belajar` | Additional attributes related to student background and learning habits. |
| —     | `id`                                                                                                                  | Unique identifier for each student.                                      |
| —     | `target`                                                                                                              | Performance level label, available only in the training set.             |

## Notes / Caveats

- FAQ sheet was public/exportable but currently mostly empty.
- Kaggle discussion topics currently returned no topics.
- Do not submit from Hermes without explicit approval from frisky.
