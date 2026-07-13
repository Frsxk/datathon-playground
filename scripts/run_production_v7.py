#!/usr/bin/env python3
"""Production v7 — v6 (interaction + temporal autocorr) PLUS:
  * denoised temporal parameterization: daily oscillation PERIOD (FFT-dominant +
    best-fit sinusoid), weekly persistence via OLS-AR(1) and Durbin-Levinson PACF(1)
    (disc_E / disc_D discovery sweep)
  * latent-gated cross-sequence coupling: skor_minat_belajar * lead-minus-lag
    daily-vs-weekly cross-correlation (disc_B); leakage-safe (target-free)
  * RANKER DIVERSITY: latent rank = z-blend of ordinal-XGB + ordinal-HistGB +
    XGB-regressor expected value (uncorrelated estimation errors partly cancel;
    attacks the per-student estimation-noise ceiling, not new signal)

Latent-score -> quartile pipeline unchanged. Seeds 42-46, seed-bagged binaries.
Writes outputs/submission_v7.csv + metrics_v7.json. NO Kaggle submission."""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import quartile_bin, SEEDS
from run_production_v6_temporal import feats as v6_feats, D, W, ORD, NC, BAG

# ---------- v7 extra features (validated survivors) ----------
def _center(M): return M - M.mean(1, keepdims=True)
def _fix(v): v = np.array(v, float); v[~np.isfinite(v)] = 0.0; return v
def _zc(v): v = _fix(v); return (v - v.mean()) / (v.std() + 1e-9)

def ar1_ols(M):
    Mc = _center(M); x0 = Mc[:, :-1]; x1 = Mc[:, 1:]
    return _fix((x0 * x1).sum(1) / ((x0 ** 2).sum(1) + 1e-9))

def _autocov(x, k):
    T = x.shape[1]; return (x[:, :T - k] * x[:, k:]).sum(1) / T
def pacf1(M):
    Mc = _center(M); r0 = _autocov(Mc, 0) + 1e-12
    return _fix(_autocov(Mc, 1) / r0)  # PACF lag1 == ACF lag1

def dom_period(M):
    Xc = _center(M); F = np.fft.rfft(Xc, axis=1); p = F.real ** 2 + F.imag ** 2
    p[:, 0] = -1.0; di = np.argmax(p, 1); L = Xc.shape[1]
    return _fix(np.where(di > 0, L / np.maximum(di, 1), 0.0))

def best_period(M, periods):
    Xc = _center(M); n, L = Xc.shape; t = np.arange(L)
    bp = np.zeros(n); bpw = np.full(n, -1.0)
    for P in periods:
        w = 2 * np.pi / P; cs = np.cos(w * t); sn = np.sin(w * t)
        a = Xc @ cs; b = Xc @ sn
        pw = a * a / (cs @ cs + 1e-12) + b * b / (sn @ sn + 1e-12)
        u = pw > bpw; bpw[u] = pw[u]; bp[u] = P
    return _fix(bp)

def _resample(M, n):
    old = np.linspace(0, 1, M.shape[1]); new = np.linspace(0, 1, n)
    return np.array([np.interp(new, old, row) for row in M])
def _ncross(A, B, off):
    if off > 0: a, b = A[:, off:], B[:, :-off]
    elif off < 0: a, b = A[:, :off], B[:, -off:]
    else: a, b = A, B
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-9
    return (a * b).sum(1) / den
def minat_x_coup(df):
    Wc = _center(df[W].to_numpy(float)); Dc = _center(df[D].to_numpy(float))
    Wr = _resample(Wc, 12); Dr = _resample(Dc, 12)
    coup = _ncross(Wr, Dr, 1) - _ncross(Wr, Dr, -1)
    return _fix(_zc(df["skor_minat_belajar"].to_numpy(float)) * _zc(coup))

_DP = np.arange(2.0, 9.0, 0.25)
def feats(df):
    X = v6_feats(df).copy()
    X["day_domperiod"]  = dom_period(df[D].to_numpy(float))
    X["day_bestperiod"] = best_period(df[D].to_numpy(float), _DP)
    X["wk_ar1_ols"]     = ar1_ols(df[W].to_numpy(float))
    X["wk_pacf1"]       = pacf1(df[W].to_numpy(float))
    X["minat_x_coup"]   = minat_x_coup(df)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---------- diverse latent-rank estimators ----------
def _z(v): v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)

def ord_xgb(Xtr, ytr, Xp, seed, bag=BAG):
    s = np.zeros(len(Xp))
    for k in range(NC - 1):
        pk = np.zeros(len(Xp))
        for bi in range(bag):
            m = XGBClassifier(**ORD, random_state=seed + 100 * bi, n_jobs=-1,
                              eval_metric="logloss", verbosity=0)
            m.fit(Xtr, (ytr > k).astype(int)); pk += m.predict_proba(Xp)[:, 1]
        s += pk / bag
    return s
def ord_hgb(Xtr, ytr, Xp, seed):
    s = np.zeros(len(Xp))
    for k in range(NC - 1):
        m = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(Xtr, (ytr > k).astype(int)); s += m.predict_proba(Xp)[:, 1]
    return s
def reg_xgb(Xtr, ytr, Xp, seed):
    m = XGBRegressor(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
                     colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5,
                     random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr); return m.predict(Xp)

def blend_rank(Xtr, ytr, Xp, seed):
    """z-blend of 3 diverse latent-rank estimators (equal weight)."""
    return _z(ord_xgb(Xtr, ytr, Xp, seed)) + _z(ord_hgb(Xtr, ytr, Xp, seed)) + _z(reg_xgb(Xtr, ytr, Xp, seed))

def main():
    tr = pd.read_csv("kaggle/train.csv"); te = pd.read_csv("kaggle/test.csv")
    sample = pd.read_csv("kaggle/sample_submission.csv")
    y = tr["target"]; X = feats(tr); Xte = feats(te)
    assert list(X.columns) == list(Xte.columns)
    print(f"nfeat={X.shape[1]} bag={BAG}")
    per_seed = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s); oof = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            oof[va_i] = blend_rank(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], s)
        per_seed.append(accuracy_score(y, quartile_bin(oof)))
        print(f"  seed {s}: {per_seed[-1]:.4f}")
    per_seed = np.array(per_seed)
    print(f"CV {per_seed.mean():.4f} +/- {per_seed.std():.4f}")

    test_scores = [pd.Series(blend_rank(X, y, Xte, s)).rank().values for s in SEEDS]
    preds = quartile_bin(np.mean(test_scores, 0)).astype(int)
    sub = pd.DataFrame({"id": te["id"], "target": preds})
    assert list(sub.columns) == ["id", "target"] and sub.shape == sample.shape
    assert sub["id"].equals(sample["id"]) and set(sub["target"]).issubset({0, 1, 2, 3})
    sub.to_csv("outputs/submission_v7.csv", index=False)
    Path("outputs/metrics_v7.json").write_text(json.dumps({
        "experiment": "v7_denoised_temporal_coupling_ranker_blend",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "bag": BAG, "feature_shape": list(X.shape),
        "added_features": ["day_domperiod", "day_bestperiod", "wk_ar1_ols", "wk_pacf1", "minat_x_coup"],
        "ranker": "z(ord_xgb)+z(ord_hgb)+z(reg_xgb)",
        "cv_mean_accuracy": float(per_seed.mean()), "cv_std": float(per_seed.std()),
        "per_seed": per_seed.round(4).tolist(),
        "prediction_counts": {str(k): int(v) for k, v in sub["target"].value_counts().sort_index().items()},
        "no_kaggle_submission_made": True}, indent=2))
    print("counts:", sub["target"].value_counts().sort_index().to_dict())
    print("wrote outputs/submission_v7.csv + metrics_v7.json")

if __name__ == "__main__":
    main()
