"""
disc_C.py — Hypothesis class C: Cross-block coupling (daily activity -> weekly grade).

Question: does daily activity DRIVE weekly grades? We aggregate the 16 daily
'aktivitas_hari_*' points into per-week-ish bins and probe lagged cross-correlation
against the 12 weekly 'nilai_minggu_*' grades and their changes. Also cross-block
coherence and daily_mean * weekly_slope style couplings.

Everything is a target-FREE row-wise transform (computable on test), center both
sequence blocks across time before any product/cross-corr, inf/nan -> 0, float.

Diagnostic: partial correlation vs v6 oof_rank (signal NOT already in v6).
INTERESTING if |partial| >= 0.04.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
WK = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]   # 12 weekly grades
DY = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]  # 16 daily activity


def clean(a):
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return a


def center_rows(M):
    """Center each row across time (subtract per-row mean over the block)."""
    M = np.asarray(M, dtype=float)
    return M - M.mean(axis=1, keepdims=True)


def zrows(M):
    """Row-wise z-score across time (for coherence-type measures)."""
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (M - mu) / sd


def row_corr(A, B):
    """Pearson corr between paired rows of A,B (already same width)."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    num = (Ac * Bc).sum(axis=1)
    den = np.sqrt((Ac**2).sum(axis=1) * (Bc**2).sum(axis=1))
    den[den < 1e-12] = 1.0
    return num / den


def build(df):
    W = clean(df[WK].values)   # (n,12)
    D = clean(df[DY].values)   # (n,16)
    n = len(df)

    Wc = center_rows(W)        # centered weekly (n,12)
    Dc = center_rows(D)        # centered daily  (n,16)

    # Bin the 16 daily points down to 12 "weeks" by simple contiguous averaging
    # so we can align daily-activity trend to weekly-grade trend row-wise.
    # 16 -> 12: overlapping windows via linear resample.
    def resample_to(M, k):
        n_, t = M.shape
        xs = np.linspace(0, t - 1, k)
        lo = np.floor(xs).astype(int)
        hi = np.clip(lo + 1, 0, t - 1)
        fr = xs - lo
        return M[:, lo] * (1 - fr) + M[:, hi] * fr

    D12 = resample_to(Dc, 12)      # centered daily resampled to 12 (n,12)
    D6 = resample_to(Dc, 6)        # coarse 6-bin daily
    W6 = resample_to(Wc, 6)

    # Also a plain non-overlapping 4-bin aggregate of daily (4 "months")
    D_raw = D  # raw daily for high-grade-week sums
    # daily aggregated to 12 raw (uncentered) for sum-in-high-weeks
    Draw12 = resample_to(D, 12)

    # weekly slope (least-squares over 12) and daily slope (over 16)
    tw = np.arange(12) - (12 - 1) / 2.0
    td = np.arange(16) - (16 - 1) / 2.0
    w_slope = (Wc * tw).sum(axis=1) / (tw**2).sum()
    d_slope = (Dc * td).sum(axis=1) / (td**2).sum()
    d_mean = D.mean(axis=1)
    w_mean = W.mean(axis=1)

    # weekly diffs (grade changes) length 11, daily diffs length 15
    Wdiff = np.diff(W, axis=1)            # (n,11) grade changes
    Ddiff = np.diff(D, axis=1)            # (n,15)

    cands = {}

    # --- (1) Row-wise cross-correlation daily-vs-weekly at several lags ---
    # Align centered daily-resampled-to-12 with centered weekly at lags -3..3.
    for lag in range(-3, 4):
        if lag == 0:
            a = D12
            b = Wc
        elif lag > 0:
            # daily leads weekly by 'lag' weeks: daily[t] vs weekly[t+lag]
            a = D12[:, :12 - lag]
            b = Wc[:, lag:]
        else:
            L = -lag
            a = D12[:, L:]
            b = Wc[:, :12 - L]
        cands[f"ccf_d2w_lag{lag:+d}"] = row_corr(a, b)

    # coarse 6-bin cross-corr at lags -2..2
    for lag in range(-2, 3):
        if lag == 0:
            a, b = D6, W6
        elif lag > 0:
            a, b = D6[:, :6 - lag], W6[:, lag:]
        else:
            L = -lag
            a, b = D6[:, L:], W6[:, :6 - L]
        cands[f"ccf6_d2w_lag{lag:+d}"] = row_corr(a, b)

    # --- (2) daily-activity DELTA -> next-week grade DELTA (lead/lag on diffs) ---
    # resample daily diffs to 11 to align with weekly diffs
    Ddiff_c = center_rows(Ddiff)
    Ddiff11 = resample_to(Ddiff_c, 11)
    Wdiff_c = center_rows(Wdiff)
    cands["ccf_ddiff_wdiff_lag0"] = row_corr(Ddiff11, Wdiff_c)
    # daily-change leads grade-change by 1 week
    cands["ccf_ddiff_wdiff_lag+1"] = row_corr(Ddiff11[:, :10], Wdiff_c[:, 1:])
    cands["ccf_ddiff_wdiff_lag-1"] = row_corr(Ddiff11[:, 1:], Wdiff_c[:, :10])

    # --- (3) sum of daily-activity in HIGH-grade weeks vs LOW-grade weeks ---
    # rank weeks by grade within student; contrast daily activity aligned to
    # high vs low grade weeks. Uses Draw12 aligned to Wc.
    order = np.argsort(Wc, axis=1)  # ascending grade weeks
    # top-3 highest grade weeks and bottom-3
    hi_idx = order[:, -3:]
    lo_idx = order[:, :3]
    rows = np.arange(n)[:, None]
    dhi = Draw12[rows, hi_idx].mean(axis=1)
    dlo = Draw12[rows, lo_idx].mean(axis=1)
    cands["dact_in_highgrade_wk"] = dhi
    cands["dact_high_minus_low_grade_wk"] = dhi - dlo
    cands["dact_high_over_low_grade_wk"] = dhi / (dlo + 1e-9)

    # --- (4) coherence: agreement of z-scored trajectories ---
    Dz12 = zrows(D12)
    Wz = zrows(Wc)
    cands["coherence_d2w"] = (Dz12 * Wz).mean(axis=1)  # mean product of z-traj
    # absolute coherence (co-movement magnitude regardless of sign)
    cands["abs_coherence_d2w"] = np.abs((Dz12 * Wz)).mean(axis=1)

    # --- (5) coupling products of block summaries ---
    cands["dmean_x_wslope"] = d_mean * w_slope
    cands["dslope_x_wslope"] = d_slope * w_slope
    cands["dslope_x_wmean"] = d_slope * w_mean
    cands["dmean_x_wmean"] = d_mean * w_mean
    # centered products (remove marginals)
    cands["dslope_x_wslope_c"] = (d_slope - d_slope.mean()) * (w_slope - w_slope.mean())

    # --- (6) does early daily activity predict late weekly grade? ---
    # first-half daily mean vs second-half weekly mean (lead coupling)
    d_first = Dc[:, :8].mean(axis=1)
    w_late = Wc[:, 6:].mean(axis=1)
    d_late = Dc[:, 8:].mean(axis=1)
    w_early = Wc[:, :6].mean(axis=1)
    cands["dfirst_x_wlate"] = d_first * w_late
    cands["dfirst_predict_wlate"] = w_late - d_first  # gap (early activity vs late grade)
    cands["cross_lead_asym"] = (d_first * w_late) - (d_late * w_early)

    # --- (7) daily-activity variability coupled to weekly persistence ---
    d_var = Dc.var(axis=1)
    w_var = Wc.var(axis=1)
    cands["dvar_x_wvar"] = d_var * w_var
    cands["dvar_over_wvar"] = d_var / (w_var + 1e-9)

    for k in cands:
        cands[k] = clean(cands[k])
    return cands


def main():
    df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
    d = np.load(f"{ROOT}/outputs/v6_oof.npz")
    cur = d["oof_rank"].astype(float)
    y = d["y"].astype(float)

    # partial-corr machinery
    ry = rankdata(y)
    ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
    tvar = np.sqrt((ty**2).mean())

    def partial(v):
        rv = rankdata(v)
        fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
        fv = np.sqrt((fr**2).mean())
        return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0

    def spear(v):
        rv = rankdata(v)
        rvc = rv - rv.mean()
        ryc = ry - ry.mean()
        den = np.sqrt((rvc**2).sum() * (ryc**2).sum())
        return float((rvc * ryc).sum() / den) if den > 1e-12 else 0.0

    cands = build(df)
    rows = []
    for name, v in cands.items():
        rows.append({"name": name, "spearman": spear(v), "partial": partial(v)})
    rows.sort(key=lambda r: -abs(r["partial"]))

    print(f"{'name':32s} {'spearman':>9s} {'partial':>9s}")
    for r in rows:
        flag = "  <== INTERESTING" if abs(r["partial"]) >= 0.04 else ""
        print(f"{r['name']:32s} {r['spearman']:>9.4f} {r['partial']:>9.4f}{flag}")

    out = {"candidates": rows, "no_kaggle_submission_made": True}
    with open(f"{ROOT}/outputs/disc_C.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote outputs/disc_C.json")


if __name__ == "__main__" and not (len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "refine"):
    main()


# ============================================================
# Refinement pass: lead-coupling windows + slope products.
# Focus on the two promising directions from pass 1:
#   dfirst_x_wlate (early daily * late weekly, partial -0.0405)
#   dslope_x_wmean / dslope_x_wslope (~0.033)
# Run:  python scripts/disc_C.py refine
# ============================================================
def refine():
    df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
    d = np.load(f"{ROOT}/outputs/v6_oof.npz")
    cur = d["oof_rank"].astype(float)
    y = d["y"].astype(float)
    ry = rankdata(y)
    ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
    tvar = np.sqrt((ty**2).mean())

    def partial(v):
        rv = rankdata(v)
        fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
        fv = np.sqrt((fr**2).mean())
        return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0

    def spear(v):
        rv = rankdata(v); rvc = rv - rv.mean(); ryc = ry - ry.mean()
        den = np.sqrt((rvc**2).sum() * (ryc**2).sum())
        return float((rvc * ryc).sum() / den) if den > 1e-12 else 0.0

    W = clean(df[WK].values); D = clean(df[DY].values)
    Wc = center_rows(W); Dc = center_rows(D)
    n = len(df)
    td = np.arange(16) - 7.5; tw = np.arange(12) - 5.5
    d_slope = (Dc * td).sum(1) / (td**2).sum()
    w_slope = (Wc * tw).sum(1) / (tw**2).sum()
    w_mean = W.mean(1); d_mean = D.mean(1)

    cands = {}
    # lead coupling: early-daily-window mean * late-weekly-window mean, many splits
    for ds in [4, 6, 8, 10]:
        for we in [6, 8, 9, 10]:
            de = Dc[:, :ds].mean(1)       # early daily (centered)
            wl = Wc[:, we:].mean(1)       # late weekly (centered)
            cands[f"dearly{ds}_x_wlate{we}"] = de * wl
    # reverse coupling (early weekly * late daily) — should be weaker if daily leads
    for ws in [4, 6]:
        for dl in [8, 10, 12]:
            we_ = Wc[:, :ws].mean(1); dl_ = Dc[:, dl:].mean(1)
            cands[f"wearly{ws}_x_dlate{dl}"] = we_ * dl_
    # lead-minus-lag asymmetry (the 'drive' direction)
    cands["lead_minus_lag_68"] = (Dc[:, :6].mean(1) * Wc[:, 8:].mean(1)) - \
                                 (Wc[:, :6].mean(1) * Dc[:, 8:].mean(1))
    # slope products & centered variants
    cands["dslope_x_wmean"] = d_slope * w_mean
    cands["dslope_x_wmean_c"] = (d_slope - d_slope.mean()) * (w_mean - w_mean.mean())
    cands["dslope_x_wslope"] = d_slope * w_slope
    cands["dmean_x_wslope_c"] = (d_mean - d_mean.mean()) * (w_slope - w_slope.mean())
    cands["dslopeSIGN_x_wslopeSIGN"] = np.sign(d_slope) * np.sign(w_slope)
    # daily slope conditioned on weekly slope sign (do they agree?)
    cands["slope_agree"] = (np.sign(d_slope) == np.sign(w_slope)).astype(float)
    # abs daily slope * abs weekly slope (co-trend magnitude)
    cands["abs_dslope_x_abs_wslope"] = np.abs(d_slope) * np.abs(w_slope)

    for k in cands: cands[k] = clean(cands[k])
    rows = [{"name": k, "spearman": spear(v), "partial": partial(v)} for k, v in cands.items()]
    rows.sort(key=lambda r: -abs(r["partial"]))
    print(f"\n=== REFINE ===\n{'name':30s} {'spearman':>9s} {'partial':>9s}")
    for r in rows:
        flag = "  <==" if abs(r["partial"]) >= 0.04 else ""
        print(f"{r['name']:30s} {r['spearman']:>9.4f} {r['partial']:>9.4f}{flag}")
    with open(f"{ROOT}/outputs/disc_C_refine.json", "w") as f:
        json.dump({"candidates": rows, "no_kaggle_submission_made": True}, f, indent=2)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "refine":
    refine()
