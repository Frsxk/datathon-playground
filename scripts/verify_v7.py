#!/usr/bin/env python3
"""v7 verification harness. Adds candidate feature blocks (discovered by the
signal-discovery workflow) on top of the exact v6 feature matrix and runs the
full seeds 42-46 ordinal CV, reporting per-seed accuracy and lift vs v6.

Edit EXTRAS below with the survivors. Each entry: name -> fn(df)->1d np.array
(target-free, computable on test). Set ACTIVE to the subset to test.

Run: .venv/Scripts/python.exe scripts/verify_v7.py            # all ACTIVE stacked
     .venv/Scripts/python.exe scripts/verify_v7.py --ablate   # leave-one-in each
NO Kaggle submission. Compute-heavy: seeds x 5 folds x ordinal(3 bins x BAG)."""
from __future__ import annotations
import sys, json, argparse, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import quartile_bin, SEEDS
from run_production_v6_temporal import feats, ord_score, D, W

# ----- candidate feature blocks (survivors from disc_* discovery sweep) -----
def _center(M): return M - M.mean(1, keepdims=True)
def _fix(v):
    v = np.array(v, float); v[~np.isfinite(v)] = 0.0; return v

def ar1_ols(M):
    """OLS AR(1) coefficient per row: regress x_t on x_{t-1} (centered)."""
    Mc = _center(M); x0 = Mc[:, :-1]; x1 = Mc[:, 1:]
    return _fix((x0 * x1).sum(1) / ((x0 ** 2).sum(1) + 1e-9))

def ols_lagk(M, k):
    """OLS lag-k coefficient (centered): denoised period-k persistence."""
    Mc = _center(M); x0 = Mc[:, :-k]; xk = Mc[:, k:]
    return _fix((x0 * xk).sum(1) / ((x0 * x0).sum(1) + 1e-9))

def _autocov(x, k):
    T = x.shape[1]; return (x[:, :T - k] * x[:, k:]).sum(1) / T

def pacf_lag(M, maxlag):
    """Durbin-Levinson PACF, per row, vectorized over rows. Returns dict lag->vec."""
    Mc = _center(M); n = Mc.shape[0]
    r = np.stack([_autocov(Mc, k) for k in range(maxlag + 1)], 1)  # (n, maxlag+1)
    r0 = r[:, 0] + 1e-12; rho = r / r0[:, None]
    pac = {k: np.zeros(n) for k in range(1, maxlag + 1)}
    phi_prev = np.zeros((n, maxlag + 1)); v = np.ones(n)
    for k in range(1, maxlag + 1):
        acc = rho[:, k].copy()
        for j in range(1, k):
            acc -= phi_prev[:, j] * rho[:, k - j]
        refl = acc / (v + 1e-12)
        phi = phi_prev.copy(); phi[:, k] = refl
        for j in range(1, k):
            phi[:, j] = phi_prev[:, j] - refl * phi_prev[:, k - j]
        v = v * (1 - refl * refl); pac[k] = _fix(refl); phi_prev = phi
    return pac

def best_period(M, periods):
    """Per-row best-fit sinusoid period (grid search maximizing projected power)."""
    Xc = _center(M); n, L = Xc.shape; t = np.arange(L)
    best_p = np.zeros(n); best_pow = np.full(n, -1.0)
    for P in periods:
        w = 2 * np.pi / P; cs = np.cos(w * t); sn = np.sin(w * t)
        a = Xc @ cs; b = Xc @ sn
        pw = a * a / (cs @ cs + 1e-12) + b * b / (sn @ sn + 1e-12)
        upd = pw > best_pow; best_pow[upd] = pw[upd]; best_p[upd] = P
    return _fix(best_p)

def dom_period(M):
    """Dominant FFT period (excluding DC), per row."""
    Xc = _center(M); F = np.fft.rfft(Xc, axis=1); p = F.real ** 2 + F.imag ** 2
    p[:, 0] = -1.0; di = np.argmax(p, 1); L = Xc.shape[1]
    return _fix(np.where(di > 0, L / np.maximum(di, 1), 0.0))

def band_ratio(M, pa, pb):
    """Power at period pa over period pb (interpolated), per row."""
    Xc = _center(M); F = np.fft.rfft(Xc, axis=1); p = F.real ** 2 + F.imag ** 2
    freqs = np.fft.rfftfreq(Xc.shape[1], d=1.0)
    def at(period):
        f0 = 1.0 / period
        for i in range(1, len(freqs)):
            if freqs[i - 1] <= f0 <= freqs[i]:
                w = (f0 - freqs[i - 1]) / (freqs[i] - freqs[i - 1] + 1e-12)
                return p[:, i - 1] * (1 - w) + p[:, i] * w
        return p[:, np.argmin(np.abs(freqs - f0))]
    return _fix(at(pa) / (at(pb) + 1e-9))

_DP = np.arange(2.0, 9.0, 0.25)  # daily period grid

def _dpac(df, k):
    return pacf_lag(df[D].to_numpy(float), 3)[k]
def _wpac(df, k):
    return pacf_lag(df[W].to_numpy(float), 3)[k]

def _resample(M, n):
    old = np.linspace(0, 1, M.shape[1]); new = np.linspace(0, 1, n)
    return np.array([np.interp(new, old, row) for row in M])
def _ncross(A, B, off):
    if off > 0: a, b = A[:, off:], B[:, :-off]
    elif off < 0: a, b = A[:, :off], B[:, -off:]
    else: a, b = A, B
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-9
    return (a * b).sum(1) / den
def _zc(v):
    v = _fix(v); return (v - v.mean()) / (v.std() + 1e-9)

def coupling(df, off_lml=True):
    """Lead-minus-lag daily-vs-weekly cross-correlation (resampled to 12), centered."""
    Wc = _center(df[W].to_numpy(float)); Dc = _center(df[D].to_numpy(float))
    Wr = _resample(Wc, 12); Dr = _resample(Dc, 12)
    if off_lml:
        return _fix(_ncross(Wr, Dr, 1) - _ncross(Wr, Dr, -1))
    return _fix(_ncross(Wr, Dr, 0))
def latent_gate(df, latent, off_lml=True):
    return _fix(_zc(df[latent].to_numpy(float)) * _zc(coupling(df, off_lml)))

EXTRAS = {
    # E: daily oscillation PERIOD (single clean scalar; spearman -0.383)
    "day_bestperiod": lambda df: best_period(df[D].to_numpy(float), _DP),
    "day_domperiod":  lambda df: dom_period(df[D].to_numpy(float)),
    "day_p3_over_p2": lambda df: band_ratio(df[D].to_numpy(float), 3.0, 2.0),
    # D: denoised period-3 daily persistence + weekly persistence estimators
    "day_pacf3":      lambda df: _dpac(df, 3),
    "day_ols_lag3":   lambda df: ols_lagk(df[D].to_numpy(float), 3),
    "wk_ar1_ols":     lambda df: ar1_ols(df[W].to_numpy(float)),
    "wk_pacf1":       lambda df: _wpac(df, 1),
    "day_ar1_ols":    lambda df: ar1_ols(df[D].to_numpy(float)),
    # B: latent-gated cross-sequence coupling (leakage-safe individual products)
    "coup_lml":         lambda df: coupling(df, True),
    "minat_x_coup":     lambda df: latent_gate(df, "skor_minat_belajar", True),
    "literasi_x_coup":  lambda df: latent_gate(df, "skor_literasi", True),
    "kedis_x_coup":     lambda df: latent_gate(df, "skor_kedisiplinan", True),
    "minat_x_coup0":    lambda df: latent_gate(df, "skor_minat_belajar", False),
    "literasi_x_coup0": lambda df: latent_gate(df, "skor_literasi", False),
}
ACTIVE = ["coup_lml", "minat_x_coup", "literasi_x_coup", "kedis_x_coup",
          "minat_x_coup0", "literasi_x_coup0"]

def build(df, active):
    X = feats(df).copy()
    for name in active:
        v = np.asarray(EXTRAS[name](df), float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        X[name] = v
    return X

def run_cv(X, y, seeds=SEEDS):
    per = []
    for s in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); oof = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            oof[va_i] = ord_score(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], s)
        per.append(accuracy_score(y, quartile_bin(oof)))
    return np.array(per)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS)); a = ap.parse_args()
    seeds = SEEDS[:a.seeds]
    tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]

    base = run_cv(feats(tr), y, seeds)
    print(f"v6 base  CV {base.mean():.4f} +/- {base.std():.4f}  {base.round(4)}")

    if a.ablate:
        for name in ACTIVE:
            p = run_cv(build(tr, [name]), y, seeds)
            print(f"  +{name:22s} CV {p.mean():.4f}  (lift {p.mean()-base.mean():+.4f})")
    else:
        p = run_cv(build(tr, ACTIVE), y, seeds)
        print(f"v7 stack CV {p.mean():.4f} +/- {p.std():.4f}  {p.round(4)}  (lift {p.mean()-base.mean():+.4f})")
        Path("outputs/verify_v7.json").write_text(json.dumps({
            "base_cv": float(base.mean()), "v7_cv": float(p.mean()),
            "lift": float(p.mean()-base.mean()), "active": ACTIVE,
            "per_seed": p.round(4).tolist(), "seeds": list(seeds),
            "no_kaggle_submission_made": True}, indent=2))
        print("wrote outputs/verify_v7.json")

if __name__ == "__main__":
    main()
