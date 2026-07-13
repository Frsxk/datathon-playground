"""disc_G.py -- Hypothesis class G: sequence distribution SHAPE & complexity.

Shape/complexity descriptors of the weekly (nilai_minggu_01..12) and daily
(aktivitas_hari_01..16) sequence blocks that mean/std/slope/autocorr miss:
skewness, kurtosis, turning points, longest monotonic run, mean-crossings,
Hurst/DFA, sample & permutation entropy, Gini of |diffs|, diff/level variance
ratio, quantile spreads.

Target-free transforms only (computable on test). Screened against v6 OOF rank
via rank partial correlation. A candidate is INTERESTING if |partial| >= 0.04.
"""
import sys
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]

# ---------------------------------------------------------------- partial screen
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
y = d["y"].astype(float)
ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def partial(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def spear(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    a = rv - rv.mean()
    b = ry - ry.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------- row descriptors
def skew_row(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1) + 1e-12
    return (((X - m) ** 3).mean(1)) / (s ** 3)


def kurt_row(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1) + 1e-12
    return (((X - m) ** 4).mean(1)) / (s ** 4) - 3.0


def turning_points(X):
    df = np.diff(X, axis=1)
    sgn = np.sign(df)
    # count sign changes in first difference
    return (np.abs(np.diff(sgn, axis=1)) > 0).sum(1).astype(float)


def longest_mono_run(X):
    # longest run of consecutive same-sign first differences
    df = np.diff(X, axis=1)
    sgn = np.sign(df)
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        best = cur_run = 1
        for j in range(1, sgn.shape[1]):
            if sgn[i, j] == sgn[i, j - 1] and sgn[i, j] != 0:
                cur_run += 1
                best = max(best, cur_run)
            else:
                cur_run = 1
        out[i] = best
    return out


def mean_crossings(X):
    c = X - X.mean(1, keepdims=True)
    s = np.sign(c)
    s[s == 0] = 1
    return (np.abs(np.diff(s, axis=1)) > 0).sum(1).astype(float)


def gini_absdiff(X):
    a = np.abs(np.diff(X, axis=1))
    a = np.sort(a, axis=1)
    n = a.shape[1]
    idx = np.arange(1, n + 1)
    num = (2 * idx - n - 1) * a
    return num.sum(1) / (n * a.sum(1) + 1e-12)


def diff_level_var_ratio(X):
    return np.var(np.diff(X, axis=1), axis=1) / (np.var(X, axis=1) + 1e-12)


def iqr_row(X):
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    return q75 - q25


def range_row(X):
    return X.max(1) - X.min(1)


def qspread_90_10(X):
    q90, q10 = np.percentile(X, [90, 10], axis=1)
    return q90 - q10


def hurst_rs(X):
    # simple rescaled-range Hurst on cumulative deviations, single scale
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i]
        n = len(x)
        m = x.mean()
        z = np.cumsum(x - m)
        R = z.max() - z.min()
        S = x.std() + 1e-12
        rs = R / S
        out[i] = np.log(rs + 1e-12) / np.log(n)
    return out


def dfa_alpha(X):
    # detrended fluctuation over 2 window scales, slope estimate
    out = np.zeros(X.shape[0])
    n = X.shape[1]
    scales = [4, n // 2] if n // 2 > 4 else [3, 5]
    for i in range(X.shape[0]):
        x = X[i]
        prof = np.cumsum(x - x.mean())
        fs = []
        for w in scales:
            nseg = len(prof) // w
            if nseg < 1:
                fs.append(np.nan)
                continue
            resid = []
            for s in range(nseg):
                seg = prof[s * w:(s + 1) * w]
                t = np.arange(w)
                cf = np.polyfit(t, seg, 1)
                resid.append(seg - np.polyval(cf, t))
            resid = np.concatenate(resid)
            fs.append(np.sqrt((resid ** 2).mean()))
        fs = np.array(fs)
        if np.any(np.isnan(fs)) or fs[0] <= 0 or fs[1] <= 0:
            out[i] = 0.0
        else:
            out[i] = (np.log(fs[1]) - np.log(fs[0])) / (
                np.log(scales[1]) - np.log(scales[0]))
    return out


def perm_entropy(X, m=3):
    # permutation entropy of ordinal patterns of length m
    perms = list(itertools.permutations(range(m)))
    pmap = {p: k for k, p in enumerate(perms)}
    out = np.zeros(X.shape[0])
    logfac = np.log(len(perms))
    for i in range(X.shape[0]):
        x = X[i]
        counts = np.zeros(len(perms))
        for j in range(len(x) - m + 1):
            pat = tuple(np.argsort(x[j:j + m], kind="stable"))
            counts[pmap[pat]] += 1
        tot = counts.sum()
        if tot == 0:
            out[i] = 0.0
            continue
        p = counts[counts > 0] / tot
        out[i] = -(p * np.log(p)).sum() / logfac
    return out


def sample_entropy(X, m=2, r_frac=0.2):
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i]
        n = len(x)
        r = r_frac * (x.std() + 1e-12)

        def phi(mm):
            cnt = 0
            tot = 0
            for a in range(n - mm):
                for b in range(a + 1, n - mm + 1):
                    tot += 1
                    if np.max(np.abs(x[a:a + mm] - x[b:b + mm])) <= r:
                        cnt += 1
            return cnt, tot

        B, tb = phi(m)
        A, ta = phi(m + 1)
        if B == 0 or A == 0:
            out[i] = 0.0
        else:
            out[i] = -np.log((A / ta) / (B / tb) + 1e-12)
    return out


def longrun_signed(X, want):
    # longest run of consecutive same-sign diffs for a specific sign (+1 up / -1 dn)
    dfx = np.diff(X, axis=1)
    sgn = np.sign(dfx)
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        best = cur_run = 0
        for j in range(sgn.shape[1]):
            if sgn[i, j] == want:
                cur_run += 1
                best = max(best, cur_run)
            else:
                cur_run = 0
        out[i] = best
    return out


def longrun_up(X):
    return longrun_signed(X, 1)


def longrun_dn(X):
    return longrun_signed(X, -1)


def longrun_asym(X):
    # up-run minus down-run: directional streakiness (shape, not level)
    return longrun_signed(X, 1) - longrun_signed(X, -1)


def bowley_skew(X):
    q75, q50, q25 = np.percentile(X, [75, 50, 25], axis=1)
    return (q75 + q25 - 2 * q50) / (q75 - q25 + 1e-12)


def hjorth_complexity(X):
    # complexity = mobility(diff) / mobility(x); mobility = sqrt(var(diff)/var)
    def mob(A):
        return np.sqrt(np.var(np.diff(A, axis=1), axis=1) / (np.var(A, axis=1) + 1e-12) + 1e-12)
    d1 = np.diff(X, axis=1)
    return mob(d1) / (mob(X) + 1e-12)


def dfa_multiscale(X):
    # proper log-log DFA slope over several scales
    n = X.shape[1]
    scales = [s for s in (3, 4, 5, 6, n // 2) if 3 <= s <= n // 2]
    scales = sorted(set(scales))
    out = np.zeros(X.shape[0])
    if len(scales) < 2:
        return out
    ls = np.log(scales)
    for i in range(X.shape[0]):
        prof = np.cumsum(X[i] - X[i].mean())
        fs = []
        for w in scales:
            nseg = len(prof) // w
            resid = []
            for s in range(nseg):
                seg = prof[s * w:(s + 1) * w]
                t = np.arange(w)
                cf = np.polyfit(t, seg, 1)
                resid.append(seg - np.polyval(cf, t))
            resid = np.concatenate(resid)
            fs.append(np.sqrt((resid ** 2).mean()) + 1e-12)
        lf = np.log(np.array(fs))
        out[i] = np.polyfit(ls, lf, 1)[0]
    return out


def above_mean_frac(X):
    # fraction of time steps spent above the row mean (shape of dwell)
    c = X - X.mean(1, keepdims=True)
    return (c > 0).mean(1).astype(float)


def max_above_streak(X):
    # longest consecutive run above the mean
    c = (X - X.mean(1, keepdims=True)) > 0
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        best = cur_run = 0
        for v in c[i]:
            cur_run = cur_run + 1 if v else 0
            best = max(best, cur_run)
        out[i] = best
    return out


def n_local_max(X):
    a = X[:, 1:-1]
    return ((a > X[:, :-2]) & (a > X[:, 2:])).sum(1).astype(float)


def _dfa_slope(X, scales, order=1):
    scales = sorted(set(s for s in scales if order + 1 <= s <= X.shape[1]))
    out = np.zeros(X.shape[0])
    if len(scales) < 2:
        return out
    ls = np.log(scales)
    for i in range(X.shape[0]):
        prof = np.cumsum(X[i] - X[i].mean())
        fs = []
        for w in scales:
            nseg = len(prof) // w
            resid = []
            for s in range(nseg):
                seg = prof[s * w:(s + 1) * w]
                t = np.arange(w)
                cf = np.polyfit(t, seg, order)
                resid.append(seg - np.polyval(cf, t))
            resid = np.concatenate(resid)
            fs.append(np.sqrt((resid ** 2).mean()) + 1e-12)
        out[i] = np.polyfit(ls, np.log(np.array(fs)), 1)[0]
    return out


def wk_dfa_scales_all(X):
    return _dfa_slope(X, [3, 4, 5, 6], order=1)


def wk_dfa_scales_fine(X):
    return _dfa_slope(X, [2, 3, 4, 6, 12], order=1)


def wk_dfa_order2(X):
    return _dfa_slope(X, [4, 5, 6], order=2)


def dfa_generic(X):
    n = X.shape[1]
    return _dfa_slope(X, [3, 4, 5, 6, 8, n // 2], order=1)


DESCRIPTORS = {
    "skew": skew_row,
    "kurt": kurt_row,
    "turnpts": turning_points,
    "longrun": longest_mono_run,
    "meancross": mean_crossings,
    "gini_absdiff": gini_absdiff,
    "diff_level_ratio": diff_level_var_ratio,
    "iqr": iqr_row,
    "range": range_row,
    "qspread_90_10": qspread_90_10,
    "hurst": hurst_rs,
    "dfa": dfa_alpha,
    "perment": perm_entropy,
    "sampent": sample_entropy,
    # --- refinement batch: complexity / run-structure variants ---
    "longrun_up": longrun_up,
    "longrun_dn": longrun_dn,
    "longrun_asym": longrun_asym,
    "bowley_skew": bowley_skew,
    "hjorth_cplx": hjorth_complexity,
    "dfa_multi": dfa_multiscale,
    "above_mean_frac": above_mean_frac,
    "max_above_streak": max_above_streak,
    "n_local_max": n_local_max,
    # --- DFA scale/order refinement ---
    "dfa_s3456": wk_dfa_scales_all,
    "dfa_fine": wk_dfa_scales_fine,
    "dfa_ord2": wk_dfa_order2,
    "dfa_gen": dfa_generic,
}


def main():
    df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
    W = df[WEEK_COLS].to_numpy(float)
    D = df[DAY_COLS].to_numpy(float)

    results = []
    for blk_name, X in [("wk", W), ("day", D)]:
        for name, fn in DESCRIPTORS.items():
            v = np.asarray(fn(X), dtype=float)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            results.append((f"{blk_name}_{name}", spear(v), partial(v)))

    results.sort(key=lambda t: -abs(t[2]))
    print(f"{'name':24s} {'spearman':>10s} {'partial':>10s}")
    print("-" * 48)
    for name, sp, pa in results:
        flag = "  <== INTERESTING" if abs(pa) >= 0.04 else ""
        print(f"{name:24s} {sp:10.4f} {pa:10.4f}{flag}")

    out = {
        "candidates": [
            {"name": n, "spearman": s, "partial": p} for n, s, p in results
        ],
        "no_kaggle_submission_made": True,
    }
    with open(f"{ROOT}/outputs/disc_G.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
