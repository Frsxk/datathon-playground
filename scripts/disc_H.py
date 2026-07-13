"""disc_H.py -- Signal discovery, hypothesis class H.

Triple/coupled interactions with KNOWN signals + segmented temporal.

Known/captured signals (must NOT be re-reported as wins):
  - interaction motiv_x_disc = skor_motivasi * skor_kedisiplinan
  - temporal lag-autocorr of daily (period ~3) and weekly (persistence rising by class)

This script builds NEW representations:
  1. motiv_x_disc coupled to temporal persistence / completion (does the
     interaction *modulate* the temporal signal, beyond either alone?)
  2. 3-way motivasi * kedisiplinan * (third latent)
  3. SEGMENTED temporal: autocorr on first-half vs second-half of each
     sequence; early-vs-late amplitude; change-in-persistence within student.
  4. segment-persistence * latent-product coupling
  5. weekly autocorr at longer lags (7-9) refined

Diagnostic: partial spearman vs v6 oof_rank. |partial| >= 0.04 => orthogonal.
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
LETTER = "H"

WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
LATENTS = [
    "skor_motivasi", "skor_kedisiplinan", "skor_literasi", "skor_minat_belajar",
    "indeks_kehadiran", "skor_ekstrakurikuler", "jarak_rumah_km", "jumlah_saudara",
]

# ---------------- partial-correlation screen ----------------
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
y = d["y"].astype(float)
ry = rankdata(y)
_ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
_tvar = np.sqrt((_ty ** 2).mean())


def clean(v):
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v


def spearman(v):
    v = clean(v)
    if np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(v), ry)[0, 1])


def partial(v):
    v = clean(v)
    if np.std(v) < 1e-12:
        return 0.0
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    if fv < 1e-9:
        return 0.0
    return float((fr * _ty).mean() / (fv * _tvar + 1e-12))


# ---------------- temporal helpers ----------------
def autocorr(mat, lag):
    """Per-row lag-k autocorrelation of a centered sequence matrix (n,T)."""
    x = mat - mat.mean(axis=1, keepdims=True)
    a = x[:, :-lag]
    b = x[:, lag:]
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-9
    return num / den


def amplitude(mat):
    """Per-row std over time of a sequence."""
    return mat.std(axis=1)


def load(path):
    df = pd.read_csv(path)
    W = df[WEEK_COLS].to_numpy(dtype=float)   # (n,12)
    D = df[DAY_COLS].to_numpy(dtype=float)    # (n,16)
    # center each sequence across time
    Wc = W - W.mean(axis=1, keepdims=True)
    Dc = D - D.mean(axis=1, keepdims=True)
    feat = {}
    mot = df["skor_motivasi"].to_numpy(float)
    dis = df["skor_kedisiplinan"].to_numpy(float)
    mxd = mot * dis                                   # KNOWN interaction
    compl = (df["tugas_selesai"] / df["tugas_diberikan"].replace(0, np.nan)).to_numpy(float)
    compl = np.nan_to_num(compl, nan=0.0)

    # --- weekly persistence measures ---
    w_ac1 = autocorr(W, 1)   # weekly lag1 persistence (known rises by class)
    w_ac2 = autocorr(W, 2)
    d_ac3 = autocorr(D, 3)   # daily period-~3 (known)

    # === (1) motiv_x_disc coupled to temporal persistence / completion ===
    feat["mxd_x_wac1"] = mxd * w_ac1
    feat["mxd_x_wac2"] = mxd * w_ac2
    feat["mxd_x_dac3"] = mxd * d_ac3
    feat["mxd_x_compl"] = mxd * compl
    feat["mxd_x_wamp"] = mxd * amplitude(W)
    feat["mxd_x_damp"] = mxd * amplitude(D)

    # === (2) 3-way motivasi*kedisiplinan*(third latent) ===
    for L in LATENTS:
        if L in ("skor_motivasi", "skor_kedisiplinan"):
            continue
        feat[f"mxd_x_{L}"] = mxd * df[L].to_numpy(float)

    # === (3) SEGMENTED temporal ===
    # weekly: first half (wk1-6) vs second half (wk7-12)
    W1, W2 = W[:, :6], W[:, 6:]
    D1, D2 = D[:, :8], D[:, 8:]
    w_ac1_h1 = autocorr(W1, 1)
    w_ac1_h2 = autocorr(W2, 1)
    d_ac3_h1 = autocorr(D1, 3)
    d_ac3_h2 = autocorr(D2, 3)
    feat["w_ac1_h1"] = w_ac1_h1
    feat["w_ac1_h2"] = w_ac1_h2
    feat["w_ac1_delta"] = w_ac1_h2 - w_ac1_h1          # change in persistence
    feat["d_ac3_h1"] = d_ac3_h1
    feat["d_ac3_h2"] = d_ac3_h2
    feat["d_ac3_delta"] = d_ac3_h2 - d_ac3_h1
    # amplitude early vs late
    w_amp1, w_amp2 = amplitude(W1), amplitude(W2)
    d_amp1, d_amp2 = amplitude(D1), amplitude(D2)
    feat["w_amp_delta"] = w_amp2 - w_amp1
    feat["w_amp_ratio"] = w_amp2 / (w_amp1 + 1e-6)
    feat["d_amp_delta"] = d_amp2 - d_amp1
    feat["d_amp_ratio"] = d_amp2 / (d_amp1 + 1e-6)
    # level drift (late mean - early mean) of centered? use raw level
    feat["w_level_drift"] = W2.mean(axis=1) - W1.mean(axis=1)
    feat["d_level_drift"] = D2.mean(axis=1) - D1.mean(axis=1)

    # === (4) segment-persistence * latent product ===
    feat["mxd_x_wac1_h2"] = mxd * w_ac1_h2
    feat["mxd_x_wac1_h1"] = mxd * w_ac1_h1
    feat["mxd_x_wac1_delta"] = mxd * (w_ac1_h2 - w_ac1_h1)
    feat["mxd_x_dac3_h2"] = mxd * d_ac3_h2
    feat["compl_x_wac1_delta"] = compl * (w_ac1_h2 - w_ac1_h1)

    # === (5) weekly autocorr longer lags 7-9 refined ===
    for k in (7, 8, 9):
        feat[f"w_ac{k}"] = autocorr(W, k)
    feat["w_ac_long_mean"] = np.mean([autocorr(W, k) for k in (7, 8, 9)], axis=0)
    feat["mxd_x_wac_long"] = mxd * feat["w_ac_long_mean"]

    return feat


def main():
    dry = "--dry-run" in sys.argv
    path = f"{ROOT}/kaggle/train.csv"
    if dry:
        # quick compile/logic check on a few rows
        df = pd.read_csv(path, nrows=int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 20)
        df.to_csv(f"{ROOT}/outputs/_disc_H_tmp.csv", index=False)
        print("dry-run ok, cols:", len(df.columns))
        return

    feat = load(path)
    rows = []
    for name, v in feat.items():
        v = clean(v)
        # align length to train
        if len(v) != len(y):
            continue
        rows.append((name, spearman(v), partial(v)))
    rows.sort(key=lambda r: -abs(r[2]))

    print(f"{'name':<24} {'spearman':>9} {'partial':>9}")
    print("-" * 46)
    for name, s, p in rows:
        flag = "  <== NEW" if abs(p) >= 0.04 else ""
        print(f"{name:<24} {s:>9.4f} {p:>9.4f}{flag}")

    out = {
        "candidates": [{"name": n, "spearman": s, "partial": p} for n, s, p in rows],
        "no_kaggle_submission_made": True,
    }
    with open(f"{ROOT}/outputs/disc_{LETTER}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote outputs/disc_{LETTER}.json ({len(rows)} candidates)")


if __name__ == "__main__":
    main()
