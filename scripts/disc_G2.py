"""disc_G2.py -- refine class G shape/complexity, round 2.

Round 1 leaders (partial): wk_dfa -0.028, day_longrun -0.024, day_dfa -0.021.
These are roughness/complexity measures that survive v6. Many high-spearman
shape stats (wk_hurst/iqr/qspread) had ~0 partial => already in v6 (they track
weekly persistence/level, the captured temporal lever).

Round 2: push the roughness/complexity direction harder --
 - proper multi-scale DFA slope (log-log regression over several window sizes)
 - Higuchi fractal dimension
 - Lempel-Ziv complexity of binarized (above/below median) sequence
 - run-length stats (max up-run, max down-run, num runs, mean run length)
 - complexity on FIRST-DIFFERENCED and on DETRENDED sequences
 - cross-block combos of the two DFA/longrun signals
 - number of local extrema normalized, "roughness" = mean|2nd diff|/std
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]

d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
y = d["y"].astype(float)
ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def partial(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def spear(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    a = rv - rv.mean()
    b = ry - ry.mean()
    dn = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / dn) if dn > 0 else 0.0


def dfa_multi(X, scales):
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i]
        prof = np.cumsum(x - x.mean())
        logn, logf = [], []
        for w in scales:
            nseg = len(prof) // w
            if nseg < 1:
                continue
            resid = []
            for s in range(nseg):
                seg = prof[s * w:(s + 1) * w]
                t = np.arange(w)
                resid.append(seg - np.polyval(np.polyfit(t, seg, 1), t))
            resid = np.concatenate(resid)
            f = np.sqrt((resid ** 2).mean())
            if f > 0:
                logn.append(np.log(w))
                logf.append(np.log(f))
        if len(logn) >= 2:
            out[i] = np.polyfit(logn, logf, 1)[0]
    return out


def higuchi_fd(X, kmax):
    out = np.zeros(X.shape[0])
    N = X.shape[1]
    for i in range(X.shape[0]):
        x = X[i]
        Lk, lk = [], []
        for k in range(1, kmax + 1):
            Lm = []
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                seg = x[idx]
                length = np.abs(np.diff(seg)).sum()
                norm = (N - 1) / (((len(idx) - 1)) * k)
                Lm.append(length * norm / k)
            if Lm:
                Lk.append(np.mean(Lm))
                lk.append(np.log(1.0 / k))
        if len(Lk) >= 2:
            Lk = np.log(np.array(Lk) + 1e-12)
            out[i] = np.polyfit(lk, Lk, 1)[0]
    return out


def lziv_binary(X):
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i]
        b = (x >= np.median(x)).astype(int)
        s = "".join(map(str, b))
        n = len(s)
        # LZ76 complexity count
        c, l, i2, k = 1, 1, 0, 1
        while True:
            if i2 + k > n:
                c += 1
                break
            if s[i2:i2 + k] != s[l:l + k] if False else s[i2 + k - 1] != s[l + k - 1]:
                if k > 1:
                    k -= 1
                i2 += 1
                if i2 == l:
                    c += 1
                    l += k
                    if l + 1 > n:
                        break
                    i2 = 0
                    k = 1
                else:
                    k = 1
            else:
                k += 1
                if l + k > n:
                    c += 1
                    break
        out[i] = c / (n / np.log2(n))
    return out


def run_stats(X, which):
    df = np.diff(X, axis=1)
    sgn = np.sign(df)
    n = X.shape[0]
    max_up = np.zeros(n); max_dn = np.zeros(n); nruns = np.zeros(n); meanrun = np.zeros(n)
    for i in range(n):
        s = sgn[i]
        runs = []
        cur_len = 1; cur_sign = s[0]
        for j in range(1, len(s)):
            if s[j] == cur_sign and s[j] != 0:
                cur_len += 1
            else:
                runs.append((cur_sign, cur_len))
                cur_sign = s[j]; cur_len = 1
        runs.append((cur_sign, cur_len))
        ups = [l for sg, l in runs if sg > 0]
        dns = [l for sg, l in runs if sg < 0]
        max_up[i] = max(ups) if ups else 0
        max_dn[i] = max(dns) if dns else 0
        nruns[i] = len(runs)
        meanrun[i] = np.mean([l for _, l in runs])
    return {"maxup": max_up, "maxdn": max_dn, "nruns": nruns, "meanrun": meanrun}[which]


def roughness(X):
    d2 = np.diff(X, n=2, axis=1)
    return np.abs(d2).mean(1) / (X.std(1) + 1e-12)


def detrend(X):
    t = np.arange(X.shape[1])
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return X - (A @ coef).T


def firstdiff(X):
    return np.diff(X, axis=1)


def local_extrema_frac(X):
    a = X[:, :-2]; b = X[:, 1:-1]; c = X[:, 2:]
    peaks = (b > a) & (b > c)
    valls = (b < a) & (b < c)
    return (peaks | valls).sum(1).astype(float) / X.shape[1]


def main():
    df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
    W = df[WEEK_COLS].to_numpy(float)
    D = df[DAY_COLS].to_numpy(float)

    feats = {}
    for bn, X in [("wk", W), ("day", D)]:
        n = X.shape[1]
        scales = [s for s in [3, 4, 5, 6, n // 2] if s <= n // 2 and s >= 2]
        scales = sorted(set(scales))
        feats[f"{bn}_dfa_ms"] = dfa_multi(X, scales)
        feats[f"{bn}_dfa_ms_diff"] = dfa_multi(firstdiff(X), [s for s in scales if s <= (n - 1) // 2])
        feats[f"{bn}_higuchi"] = higuchi_fd(X, max(2, n // 3))
        feats[f"{bn}_higuchi_dt"] = higuchi_fd(detrend(X), max(2, n // 3))
        feats[f"{bn}_lziv"] = lziv_binary(X)
        feats[f"{bn}_lziv_diff"] = lziv_binary(firstdiff(X))
        feats[f"{bn}_maxup"] = run_stats(X, "maxup")
        feats[f"{bn}_maxdn"] = run_stats(X, "maxdn")
        feats[f"{bn}_nruns"] = run_stats(X, "nruns")
        feats[f"{bn}_meanrun"] = run_stats(X, "meanrun")
        feats[f"{bn}_rough"] = roughness(X)
        feats[f"{bn}_rough_dt"] = roughness(detrend(X))
        feats[f"{bn}_extrema"] = local_extrema_frac(X)

    # cross-block combos of the surviving roughness signals
    feats["dfa_wk_plus_day"] = (
        rankdata(feats["wk_dfa_ms"]) + rankdata(feats["day_dfa_ms"]))
    feats["higuchi_wk_plus_day"] = (
        rankdata(feats["wk_higuchi"]) + rankdata(feats["day_higuchi"]))
    feats["rough_wk_plus_day"] = (
        rankdata(feats["wk_rough"]) + rankdata(feats["day_rough"]))

    results = []
    for name, v in feats.items():
        v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
        results.append((name, spear(v), partial(v)))
    results.sort(key=lambda t: -abs(t[2]))

    print(f"{'name':24s} {'spearman':>10s} {'partial':>10s}")
    print("-" * 48)
    for name, sp, pa in results:
        flag = "  <== INTERESTING" if abs(pa) >= 0.04 else ""
        print(f"{name:24s} {sp:10.4f} {pa:10.4f}{flag}")

    out = {"candidates": [{"name": n, "spearman": s, "partial": p}
                          for n, s, p in results],
           "no_kaggle_submission_made": True}
    with open(f"{ROOT}/outputs/disc_G2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
