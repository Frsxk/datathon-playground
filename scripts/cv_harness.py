#!/usr/bin/env python3
"""Reusable latent-score -> quartile CV harness for the Datathon playground.

Replicates the run_production_v3 evaluation protocol exactly so numbers are
comparable to the documented 0.5286 baseline:
  * per seed: 5-fold OOF latent score (estimators fit in-fold)
  * z-score OOF, blend, quartile_bin the full OOF vector, accuracy vs y
  * average over seeds 42..46

Usage: import build_features / cv_latent, or run directly for the baseline table.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle"
SEEDS = (42, 43, 44, 45, 46)
NC = 4
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]


def quartile_bin(s):
    r = pd.Series(s).rank(method="first")
    return np.clip((r / (len(s) + 1) * NC).astype(int), 0, NC - 1).values


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def ev(p):
    return (p * np.arange(p.shape[1])).sum(1)


def _seq(out, src, cols, pfx):
    arr = src[cols].to_numpy(float)
    h = len(cols) // 2
    out[f"{pfx}_mean"] = arr.mean(1)
    out[f"{pfx}_std"] = arr.std(1)
    out[f"{pfx}_min"] = arr.min(1)
    out[f"{pfx}_max"] = arr.max(1)
    out[f"{pfx}_range"] = arr.max(1) - arr.min(1)
    out[f"{pfx}_median"] = np.median(arr, 1)
    out[f"{pfx}_iqr"] = np.percentile(arr, 75, 1) - np.percentile(arr, 25, 1)
    out[f"{pfx}_last_minus_first"] = arr[:, -1] - arr[:, 0]
    out[f"{pfx}_late_minus_early"] = arr[:, h:].mean(1) - arr[:, :h].mean(1)
    d = np.diff(arr, 1)
    out[f"{pfx}_diff_std"] = d.std(1)
    out[f"{pfx}_diff_abs_mean"] = np.abs(d).mean(1)
    out[f"{pfx}_pos_steps"] = (d > 0).sum(1)
    out[f"{pfx}_neg_steps"] = (d < 0).sum(1)
    xc = np.arange(arr.shape[1]) - (arr.shape[1] - 1) / 2
    out[f"{pfx}_slope"] = (arr - arr.mean(1, keepdims=True)) @ xc / (xc**2).sum()


def build_features(df, kind="full"):
    """kind in {full, lean, signal}."""
    Wm = df[W].to_numpy(float)
    cum = np.cumsum(Wm, 1)
    out = pd.DataFrame(index=df.index)
    cr = (df["tugas_selesai"] / df["tugas_diberikan"].replace(0, np.nan)).fillna(0)

    # --- signal core (always) ---
    out["compl_ratio"] = cr
    out["compl_diff"] = df["tugas_selesai"] - df["tugas_diberikan"]
    out["compl_remaining"] = df["tugas_diberikan"] - df["tugas_selesai"]
    out["minggu_std"] = Wm.std(1)
    out["minggu_range"] = Wm.max(1) - Wm.min(1)
    out["minggu_iqr"] = np.percentile(Wm, 75, 1) - np.percentile(Wm, 25, 1)
    out["abs_change_sum"] = np.abs(Wm).sum(1)
    out["abs_change_mean"] = np.abs(Wm).mean(1)
    out["cum_range"] = cum.max(1) - cum.min(1)
    out["cum_std"] = cum.std(1)
    out["cum_max"] = cum.max(1)
    out["cum_min"] = cum.min(1)
    out["skor_tryout"] = df["skor_tryout"]
    # interactions among signal
    out["vol_x_compl"] = out["minggu_std"] * cr
    out["range_x_compl"] = out["minggu_range"] * cr
    out["tryout_x_compl"] = df["skor_tryout"] * cr
    out["abschg_x_compl"] = out["abs_change_sum"] * cr

    if kind == "signal":
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- lean: add richer weekly sequence summaries (still no noise cols) ---
    _seq(out, df, W, "minggu")
    out["cum_std2"] = cum.std(1)
    if kind == "lean":
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- full: add daily activity + misc raw (the noisy stuff) ---
    _seq(out, df, Dc, "hari")
    for c in ["skor_motivasi", "skor_kedisiplinan", "jarak_rumah_km",
              "skor_ekstrakurikuler", "indeks_kehadiran", "skor_literasi",
              "jumlah_saudara", "skor_minat_belajar", "urutan_ujian"]:
        out[c] = df[c]
    cc = df["kelas"].value_counts()
    out["kelas_freq"] = df["kelas"].map(cc).astype(float)
    out["kelas_mod10"] = df["kelas"] % 10
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cv_latent(X, y, estimators, seeds=SEEDS, blend="mean_z"):
    """estimators: dict name -> (factory(seed), kind) kind in {reg,ev}.
    Returns per-seed accuracies of the equal-weight z-score blend."""
    accs, all_oof = [], {}
    for s in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = {k: np.zeros(len(y)) for k in estimators}
        for tr, va in cv.split(X, y):
            Xtr, Xva, ytr = X.iloc[tr], X.iloc[va], y.iloc[tr]
            for name, (fac, kind) in estimators.items():
                m = fac(s); m.fit(Xtr, ytr)
                oof[name][va] = m.predict(Xva) if kind == "reg" else ev(m.predict_proba(Xva))
        Z = {k: z(v) for k, v in oof.items()}
        score = np.mean([Z[k] for k in Z], 0)
        accs.append(accuracy_score(y, quartile_bin(score)))
        all_oof[s] = Z
    return np.array(accs), all_oof


if __name__ == "__main__":
    from xgboost import XGBRegressor, XGBClassifier
    tr = pd.read_csv(DATA / "train.csv"); y = tr["target"]
    def xreg(s): return XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
        random_state=s, n_jobs=-1)
    def xclf(s): return XGBClassifier(objective="multi:softprob", num_class=4,
        n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.9,
        colsample_bytree=0.9, random_state=s, n_jobs=-1, eval_metric="mlogloss", verbosity=0)
    est = {"xgb_reg": (xreg, "reg"), "xgb_ev": (xclf, "ev")}
    for kind in ("full", "lean", "signal"):
        X = build_features(tr, kind)
        accs, _ = cv_latent(X, y, est)
        print(f"{kind:7s} nfeat={X.shape[1]:3d}  CV {accs.mean():.4f} +/- {accs.std():.4f}  {accs.round(4)}")
