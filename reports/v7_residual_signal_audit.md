# v7 Residual Signal Audit

Generated at: `2026-07-13T03:09:26.542297+00:00`

## Safety

- Read-only audit; no model fitting, synthetic generation, threshold changes, or Kaggle calls.
- OOF scores came from `outputs/v8_v7_scores.npz`.
- Features are target-free; labels are used only for residual/error analysis.

## v7 OOF summary

- Seeds: `[42, 43, 44, 45, 46]`
- Per-seed accuracy: `[0.621563, 0.613437, 0.62625, 0.62625, 0.623437]`
- Mean accuracy: `0.622188`
- Mean-score accuracy: `0.627812`
- Mean boundary distance: `0.092863`

### Confusion matrix

```text
[[597, 180, 35, 1], [173, 408, 188, 27], [29, 181, 403, 171], [1, 31, 174, 601]]
```

## Top stable residual signals

| feature | signed rho | abs rho | error rho | seed sign agreement | train/test SMD | risk |
|---|---:|---:|---:|---:|---:|---|
| fe__abschg_x_compl | -0.2476 | -0.0001 | -0.0126 | 1.00 | 0.009 | low |
| v7__abschg_x_compl | -0.2476 | -0.0001 | -0.0126 | 1.00 | 0.009 | low |
| fe__vol_x_compl | -0.2474 | -0.0004 | -0.0123 | 1.00 | 0.015 | low |
| v7__vol_x_compl | -0.2474 | -0.0004 | -0.0123 | 1.00 | 0.015 | low |
| fe__range_x_compl | -0.2404 | 0.0072 | -0.0037 | 1.00 | 0.015 | low |
| v7__range_x_compl | -0.2404 | 0.0072 | -0.0037 | 1.00 | 0.015 | low |
| fe__minggu_std | -0.1791 | -0.0221 | -0.0322 | 1.00 | 0.033 | low |
| v7__minggu_std | -0.1791 | -0.0221 | -0.0322 | 1.00 | 0.033 | low |
| fe__abs_change_mean | -0.1769 | -0.0192 | -0.0308 | 1.00 | 0.023 | low |
| v7__abs_change_mean | -0.1769 | -0.0192 | -0.0308 | 1.00 | 0.023 | low |
| fe__abs_change_sum | -0.1769 | -0.0192 | -0.0308 | 1.00 | 0.023 | low |
| v7__abs_change_sum | -0.1769 | -0.0192 | -0.0308 | 1.00 | 0.023 | low |
| fe__minggu_range | -0.1738 | -0.0115 | -0.0206 | 1.00 | 0.038 | low |
| v7__minggu_range | -0.1738 | -0.0115 | -0.0206 | 1.00 | 0.038 | low |
| fe__minggu_iqr | -0.1728 | -0.0299 | -0.0393 | 1.00 | 0.017 | low |
| v7__minggu_iqr | -0.1728 | -0.0299 | -0.0393 | 1.00 | 0.017 | low |
| fe__tryout_x_compl | -0.1692 | 0.0202 | 0.0111 | 1.00 | -0.044 | low |
| v7__tryout_x_compl | -0.1692 | 0.0202 | 0.0111 | 1.00 | -0.044 | low |
| fe__cum_std | -0.1693 | -0.0248 | -0.0377 | 1.00 | 0.018 | low |
| fe__cum_std2 | -0.1693 | -0.0248 | -0.0377 | 1.00 | 0.018 | low |

## Interpretation guardrails

- A residual correlation is a discovery lead, not proof of causal signal.
- Promote only features with stable direction across seeds and low train/test shift.
- Any next model experiment must preserve v7's exact 200/200/200/200 submission balancing.
