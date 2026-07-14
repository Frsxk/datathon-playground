# Advanced Generator Fingerprint Audit

Generated: `2026-07-14T11:11:12.534526+00:00`

## Safety

Read-only local audit. No Kaggle submission was made, and no test labels were inferred.

## Top monotone generator-parameter features

| feature | quartile acc | spearman |
|---|---:|---:|
| completion_ratio | 0.381563 | +0.4028 |
| tryout_x_completion | 0.373437 | +0.4098 |
| wk_std | 0.362187 | +0.3969 |
| wk_range | 0.351562 | +0.3780 |
| motiv_x_disc | 0.344687 | +0.3545 |
| wk_entropy | 0.326250 | +0.3096 |
| day_longest_high_run | 0.325937 | -0.2672 |
| completion_remaining | 0.318750 | -0.2643 |
| wk_longest_positive_run | 0.316875 | +0.2719 |
| wk_longest_negative_run | 0.311250 | +0.2655 |
| exam_order_x_completion | 0.307500 | +0.2211 |
| wk_period4 | 0.306875 | +0.2461 |
| tugas_selesai | 0.305937 | +0.2109 |
| day_period4 | 0.288438 | +0.1361 |
| day_period5 | 0.283438 | -0.1260 |
| day_period3 | 0.276250 | +0.1140 |
| skor_tryout | 0.274687 | +0.1076 |
| skor_kedisiplinan | 0.274062 | -0.0007 |
| skor_motivasi | 0.271250 | +0.0128 |
| wk_period3 | 0.270937 | +0.0430 |

## Top unsupervised cluster-majority checks

| rep | transform | k | CV acc | full train purity |
|---|---|---:|---:|---:|
| generator_params | none | 32 | 0.416250 | 0.424063 |
| generator_params | pca8 | 64 | 0.412187 | 0.474687 |
| generator_params | none | 8 | 0.410938 | 0.415312 |
| generator_params | none | 64 | 0.409687 | 0.471250 |
| generator_params | pca8 | 32 | 0.408438 | 0.445312 |
| generator_params | pca8 | 16 | 0.392500 | 0.425312 |
| generator_params | none | 16 | 0.383125 | 0.407187 |
| raw | none | 32 | 0.381875 | 0.408750 |
| generator_params | none | 4 | 0.381250 | 0.378750 |
| raw | none | 64 | 0.368750 | 0.420938 |
| generator_params | pca8 | 4 | 0.368437 | 0.378125 |
| raw | none | 4 | 0.356875 | 0.361875 |
| raw | none | 16 | 0.356875 | 0.370312 |
| raw | pca8 | 16 | 0.356563 | 0.362187 |
| raw | pca8 | 32 | 0.349687 | 0.398750 |
| raw | pca8 | 8 | 0.349062 | 0.341875 |
| raw | pca8 | 64 | 0.341875 | 0.421875 |
| raw | none | 8 | 0.339375 | 0.338438 |
| generator_params | pca8 | 8 | 0.332188 | 0.400000 |
| raw | pca8 | 4 | 0.258437 | 0.363125 |

## Quantization / grid notes

| feature | unique combined | decimals | min step | integer grid |
|---|---:|---:|---:|---|
| id | 4000 | 0 | 1 | True |
| kelas | 795 | 0 | 1 | True |
| tugas_diberikan | 113 | 0 | 1 | True |
| tugas_selesai | 107 | 0 | 1 | True |
| aktivitas_hari_03 | 653 | 1 | 0.1 | False |
| aktivitas_hari_16 | 650 | 1 | 0.1 | False |
| aktivitas_hari_09 | 647 | 1 | 0.1 | False |
| aktivitas_hari_05 | 641 | 1 | 0.1 | False |
| aktivitas_hari_08 | 641 | 1 | 0.1 | False |
| aktivitas_hari_12 | 641 | 1 | 0.1 | False |
| aktivitas_hari_04 | 640 | 1 | 0.1 | False |
| aktivitas_hari_13 | 640 | 1 | 0.1 | False |
| aktivitas_hari_06 | 638 | 1 | 0.1 | False |
| aktivitas_hari_01 | 637 | 1 | 0.1 | False |
| aktivitas_hari_07 | 637 | 1 | 0.1 | False |
| aktivitas_hari_10 | 635 | 1 | 0.1 | False |
| aktivitas_hari_11 | 633 | 1 | 0.1 | False |
| aktivitas_hari_14 | 631 | 1 | 0.1 | False |
| aktivitas_hari_15 | 631 | 1 | 0.1 | False |
| aktivitas_hari_02 | 627 | 1 | 0.1 | False |
| skor_tryout | 542 | 1 | 0.1 | False |
| nilai_minggu_03 | 238 | 1 | 0.1 | False |
| nilai_minggu_08 | 238 | 1 | 0.1 | False |
| nilai_minggu_11 | 237 | 1 | 0.1 | False |
| nilai_minggu_06 | 236 | 1 | 0.1 | False |

## Interpretation

No advanced local fingerprint explains a 0.82 public score. Single formulae and unsupervised clusters are far below v7, so further local climb is unlikely without an external clue or leaderboard probing.
