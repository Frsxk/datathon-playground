# Datathon Leakage and Hidden-Pattern Audit

Generated at: `2026-07-14T10:01:54.192214+00:00`

## Safety

- Read-only audit. No Kaggle submission was made.
- No test labels are inferred or used; this checks structural clues only.

## Exact overlap / duplicates

```json
{
  "overlap_count": 0,
  "test_overlap_count": 0,
  "train_duplicate_feature_rows": 0,
  "test_duplicate_feature_rows": 0
}
```

## Nearest-neighbor summary

```json
{
  "min_distance": 3.9575720514238952,
  "p01_distance": 4.315853582992312,
  "p05_distance": 4.5662869027754605,
  "median_distance": 5.2283471064104665,
  "nearest_target_counts": {
    "0": 203,
    "1": 221,
    "2": 193,
    "3": 183
  },
  "near_duplicate_threshold_count_lt_1e_minus_6": 0
}
```

## Top train/test shifts

| feature | SMD |
|---|---:|
| aktivitas_hari_09 | 0.0767 |
| aktivitas_hari_06 | -0.0762 |
| nilai_minggu_05 | -0.0552 |
| nilai_minggu_07 | 0.0528 |
| nilai_minggu_03 | -0.0512 |
| aktivitas_hari_05 | -0.0489 |
| skor_minat_belajar | 0.0482 |
| aktivitas_hari_15 | -0.0467 |
| skor_ekstrakurikuler | -0.0444 |
| skor_tryout | -0.0438 |
| aktivitas_hari_11 | 0.0382 |
| nilai_minggu_04 | -0.0379 |
| nilai_minggu_06 | -0.0374 |
| nilai_minggu_12 | 0.0363 |
| nilai_minggu_09 | 0.0342 |
| aktivitas_hari_04 | 0.0335 |
| aktivitas_hari_03 | 0.0329 |
| aktivitas_hari_07 | -0.0316 |
| tugas_selesai | 0.0289 |
| nilai_minggu_11 | 0.0285 |

## Top local hidden-pattern feature-set checks

| feature set | n | HGB OOF accuracy |
|---|---:|---:|
| full_raw | 41 | 0.474687 |
| scalar_nonseq | 13 | 0.391250 |
| weekly_only | 12 | 0.334688 |
| daily_only | 16 | 0.316563 |
| id_patterns_m17 | 4 | 0.274062 |
| id_patterns_m32+kelas_urutan | 6 | 0.272813 |
| id_patterns_m32 | 4 | 0.268437 |
| id_patterns_m20+kelas_urutan | 6 | 0.266562 |
| id_patterns_m17+kelas_urutan | 6 | 0.265625 |
| id_patterns_m5 | 4 | 0.265313 |
| id_patterns_m64 | 4 | 0.265313 |
| id_patterns_m9+kelas_urutan | 6 | 0.264688 |
| id_patterns_m40+kelas_urutan | 6 | 0.264375 |
| kelas_urutan | 2 | 0.264062 |
| id_patterns_m2+kelas_urutan | 6 | 0.263125 |
| id_patterns_m8 | 4 | 0.262813 |
| id_patterns_m15+kelas_urutan | 6 | 0.262813 |
| id_patterns_m4 | 4 | 0.262188 |
| id_patterns_m5+kelas_urutan | 6 | 0.261875 |
| id_patterns_m24 | 4 | 0.261875 |
| id_patterns_m3 | 4 | 0.260937 |
| id_patterns_m8+kelas_urutan | 6 | 0.260937 |
| id_patterns_m18+kelas_urutan | 6 | 0.260000 |
| id_patterns_m10 | 4 | 0.259375 |
| id_patterns_m11+kelas_urutan | 6 | 0.259375 |

## Initial interpretation

No obvious row-order/ID/admin shortcut appears in local CV. The 0.82 public score likely requires a deeper feature-generation rule, external clue, or public-LB probing rather than a simple deterministic ID leak.
