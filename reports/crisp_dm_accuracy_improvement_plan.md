# Datathon Playground Accuracy Improvement Plan

> **For Hermes:** This is a planning and scaffolding artifact only. Do not run model training, cross-validation, Optuna, Kaggle submission, or long feature generation unless frisky explicitly approves a specific experiment.

**Goal:** Improve the Datathon Playground student-performance classifier beyond the current best local CV of `0.5285625` and public LB tie of `0.57916` while reducing leaderboard overfitting risk.

**Architecture:** Follow CRISP-DM end-to-end: clarify business objective and metric alignment, diagnose data/label structure, prepare leakage-safe feature variants, design ordinal/latent-score models, evaluate with stable repeated CV and public-LB calibration, and deploy only validated submission candidates. All new experiments must be queued in `outputs/experiment_backlog.json` and run through a guarded runner so compute-heavy work only happens after explicit approval.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, XGBoost, Optuna, Kaggle CLI, existing scripts under `scripts/`.

---

## Current baseline and constraints

- Current best local production v3 CV: `0.5285625`.
- Current best public LB: `0.57916`, tied by `submission_production_v2.csv` and `submission_production_v3.csv`.
- Data shape: train `3200 x 43`, test `800 x 42`.
- Target: ordered 4-class labels `0..3`.
- Metric: accuracy.
- Winning local framing so far: latent continuous score + quartile binning into balanced `200/200/200/200` test predictions.
- Key risk: local CV gains do not necessarily improve public LB because public test labels may reward the same coarse ranking as v2, or the public split may differ from repeated CV.
- Hard guardrail: no Kaggle submission without explicit approval.

---

## CRISP-DM phase 1: Business understanding

### Objective

Maximize public/private leaderboard accuracy while preserving robust local validation. Since this is a learning playground, the secondary objective is to generate reusable methods for the real Datathon.

### Success criteria

1. Local repeated-CV score improves by at least `+0.002` over `0.5285625`.
2. Validation is stable across seeds: no single lucky seed dominates.
3. Public LB improvement over `0.57916`, or if tied, better evidence that the submission is robust for private score.
4. Submission remains valid: `id,target`, 800 rows, labels `0..3`.

### Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Public LB overfit | Test public split is small/noisy | Treat public LB as confirmation only; prioritize repeated CV and perturbation tests |
| Quartile assumption wrong on private | Forced 200/class may overfit public distribution | Add experiments with learned thresholds and near-quartile thresholds |
| Leakage from class/category encoding | `kelas` can accidentally leak target if encoded globally | Only fold-safe encoders; avoid current poor target encoding unless redesigned |
| Compute waste | Full Optuna took many hours | Use staged experiments: static checks -> smoke (`1 seed`) -> full repeated CV only after approval |

### Deliverables

- `reports/crisp_dm_accuracy_improvement_plan.md` — this plan.
- `outputs/experiment_backlog.json` — prioritized experiment queue.
- `scripts/feature_factory_v4.py` — feature recipes only, no training.
- `scripts/experiment_backlog.py` — safe backlog listing/dry-run CLI.

---

## CRISP-DM phase 2: Data understanding

### Known signal

High-signal families from prior EDA:

- Assignment completion: `tugas_completion_ratio`, `tugas_completion_gap`, `tugas_remaining`.
- Weekly score volatility: `minggu_std`, `minggu_iqr`, `minggu_range`, quantiles.
- Weekly/daily sequence dynamics: slopes, positive/negative steps, diff volatility.
- Ordinal framing: labels behave like binned latent academic performance.

### New diagnostics to run later

These are cheap-to-medium diagnostics but should still be scheduled explicitly:

1. **Per-class profile table**
   - Compare class-wise medians/IQR for all engineered feature families.
   - Purpose: identify monotonic vs non-monotonic features.

2. **OOF error audit**
   - For current v3 OOF predictions, identify systematic confusions: `0<->1`, `1<->2`, `2<->3`.
   - Purpose: build boundary-specific features.

3. **Rank stability audit**
   - Compare v2 vs v3 latent test ranks, especially 60 rows whose labels changed.
   - Purpose: understand why public LB tied despite 60 changed labels.

4. **Train/test shift check**
   - Compare distributions of key features between train and test.
   - Purpose: avoid features with strong covariate shift.

5. **Pseudo-label caution report**
   - Only consider if train/test shift is mild and public LB feedback supports it.
   - Purpose: avoid unsafe leaderboard overfitting.

---

## CRISP-DM phase 3: Data preparation

### Feature strategy

Build feature recipes as composable, deterministic functions. Never use target information except inside fold-local transforms.

#### Recipe A: boundary-aware ordinal features

Purpose: improve adjacent class boundaries.

Candidate features:

- `completion_shortfall_x_minggu_volatility`
- `tryout_relative_to_weekly_mean`
- `weekly_last_quarter_mean - weekly_first_quarter_mean`
- `weekly_midterm_recovery`: late mean minus minimum early/mid score
- `activity_spike_count`: daily activity jumps above row median + row IQR

#### Recipe B: robust sequence summaries

Purpose: reduce sensitivity to noisy single columns.

Candidate features:

- Trimmed mean for weekly scores and daily activity.
- Median absolute deviation.
- Longest increasing/decreasing run.
- Number of sign changes in weekly differences.
- Ratio of late volatility to early volatility.

#### Recipe C: threshold and rank features

Purpose: support latent-score ranking and threshold learning.

Candidate features:

- Row percentile ranks of key raw/engineered features within train+test combined, target-free.
- Normal scores for monotonic features.
- Rank-averaged composite academic score from stable features.

#### Recipe D: safe `kelas` encodings

Current target encoding was poor. Safer variants:

- Frequency only.
- Hash buckets/mod buckets.
- Leave-one-fold target encoding with smoothing and nested CV only, if revisited.
- Interaction: `kelas_frequency x completion_ratio`.

### Preparation acceptance criteria

- Feature factory can run in dry-run/sample mode without training.
- Feature names are deterministic and documented.
- No target leakage in recipe functions.
- Any fold-dependent transform must expose an explicit `fit_transform_fold` API later.

---

## CRISP-DM phase 4: Modeling

### Model families to try later

1. **Latent score + threshold search**
   - Keep current regressor/classifier expected-value blend.
   - Replace fixed quartile binning with threshold search on OOF scores.
   - Test three threshold modes:
     - fixed quartile,
     - learned global thresholds,
     - constrained near-quartile thresholds.

2. **Ordinal decomposition**
   - Train binary classifiers for `target > 0`, `target > 1`, `target > 2`.
   - Convert cumulative probabilities into expected value or class prediction.
   - Advantage: directly models ordered boundaries.

3. **Pairwise adjacent boundary specialists**
   - Specialist models for `0 vs 1`, `1 vs 2`, `2 vs 3`.
   - Use only to adjust uncertain examples near thresholds.

4. **Rank ensemble**
   - Blend latent score ranks from XGB regressor, XGB EV, HGB EV, Ridge, logistic EV.
   - Optimize ranks/weights, not raw scores, to reduce calibration sensitivity.

5. **Seed-bagged small ensemble**
   - Keep model complexity stable.
   - Focus on seed stability and rank consistency rather than max one-seed CV.

### Modeling guardrails

- Every model experiment must specify expected compute cost.
- Start with `smoke` mode: one seed, fewer estimators, no Optuna.
- Full repeated CV requires explicit frisky approval.
- Kaggle submission requires separate explicit approval.

---

## CRISP-DM phase 5: Evaluation

### Local validation protocol

Primary:

- Repeated StratifiedKFold with seeds `42..46`.
- Report mean, std, and per-seed scores.

Secondary:

- Boundary accuracy by adjacent label pair.
- Macro recall by class.
- OOF confusion matrix.
- Test prediction class balance.
- Rank correlation vs v2/v3.

### Candidate promotion rules

Promote to submission candidate only if:

1. Mean CV beats `0.5285625` by at least `0.002`, or it ties but materially improves stability.
2. No validation seed is catastrophically worse than v3.
3. Prediction distribution is plausible, not accidental collapse.
4. Train/test shift diagnostics do not flag the feature set as unstable.
5. The candidate is reproducible from committed scripts/config.

### Public LB interpretation

If public LB ties again despite local gain:

- Do not assume failure; public split may be saturated by rank/quartile structure.
- Compare changed predictions with prior submissions and focus on private-score robustness.
- Prefer diverse candidates only when they improve a different error slice.

---

## CRISP-DM phase 6: Deployment/submission

### Submission workflow

1. Generate candidate CSV locally.
2. Validate shape, columns, id order, target labels, counts.
3. Show exact file and Kaggle command to frisky.
4. Wait for explicit approval.
5. Submit with descriptive message.
6. Fetch submissions and leaderboard to verify status/score.
7. Record result in a report JSON/MD.

### Artifact policy

Commit:

- Plan docs.
- Experiment configs/backlogs.
- Reusable feature/model code.
- Final result JSONs worth preserving.

Do not commit by default:

- Huge logs.
- Temporary watchdog files.
- One-off process wrappers unless they become reusable.

---

## Prioritized next experiments

### P0 — analysis-only, no training

1. Compare v2/v3 submissions and identify changed rows.
2. Build feature-factory dry-run summary on first N rows.
3. Build OOF-error audit script interface but do not execute until OOF data exists.

### P1 — cheap smoke experiments later

1. Boundary-aware feature smoke with one seed and reduced estimators.
2. Threshold-search smoke using existing v3 OOF code path.
3. Rank-ensemble smoke with existing estimators but one seed.

### P2 — full experiments later

1. Full repeated CV for best P1 candidates.
2. Full threshold optimization with seed-bagging.
3. Conservative Optuna around the new best params, narrowed search space.

### P3 — submission candidates later

1. Best local-CV candidate.
2. Most stable/private-robust candidate.
3. Most diverse candidate with similar CV but different changed rows.

---

## Implementation tasks already scaffolded

### Task 1: Create experiment backlog

**File:** `outputs/experiment_backlog.json`

Defines candidate experiments with CRISP-DM phase, hypothesis, cost, risk, required approval, and command template.

### Task 2: Create feature factory scaffold

**File:** `scripts/feature_factory_v4.py`

Provides deterministic feature recipe functions and a `--dry-run --rows N` mode. It must not train models.

### Task 3: Create backlog CLI

**File:** `scripts/experiment_backlog.py`

Lists experiments and prints planned commands. It must not execute training commands unless future code adds an explicit approval mechanism.

### Task 4: Validate without compute

Run only:

```bash
.venv/bin/python -m py_compile scripts/feature_factory_v4.py scripts/experiment_backlog.py
.venv/bin/python scripts/experiment_backlog.py --list
.venv/bin/python scripts/feature_factory_v4.py --dry-run --rows 5
```

These checks are allowed because they do not train/tune/CV.
