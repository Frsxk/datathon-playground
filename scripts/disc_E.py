"""Discovery batch E: Daily/weekly oscillation PHASE / PERIOD / spectral shape.

v6 captured autocorr AMPLITUDE of the period-~3 daily oscillation. Hypothesis:
the target may instead (or additionally) control the PERIOD, PHASE, or spectral
SHAPE of the oscillation. This batch computes FFT-based and sign-based features
of the (time-centered) daily aktivitas_hari_01..16 and weekly nilai_minggu_01..12
series and screens each for signal ORTHOGONAL to the v6 latent rank via
rank partial correlation.

All transforms are target-free (computable on test). inf/nan -> 0.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"

# ---------- partial-correlation screen ----------
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
y = d["y"].astype(float)
ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def spear(v):
    v = np.asarray(v, float)
    if np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(v), ry)[0, 1])


def partial(v):
    v = np.asarray(v, float)
    if np.std(v) < 1e-12:
        return 0.0
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def clean(a):
    a = np.asarray(a, float)
    a[~np.isfinite(a)] = 0.0
    return a


# ---------- load data ----------
tr = pd.read_csv(f"{ROOT}/kaggle/train.csv")
DAY = [c for c in tr.columns if c.startswith("aktivitas_hari_")]
WK = [c for c in tr.columns if c.startswith("nilai_minggu_")]
DAY.sort()
WK.sort()

Draw = tr[DAY].to_numpy(float)   # (n,16)
Wraw = tr[WK].to_numpy(float)    # (n,12)

# time-center each series (subtract per-row temporal mean)
Dc = Draw - Draw.mean(axis=1, keepdims=True)
Wc = Wraw - Wraw.mean(axis=1, keepdims=True)


# ---------- spectral helpers ----------
def rfft_power(X):
    """Return (freqs, power) for real FFT along axis=1. power shape (n, F)."""
    F = np.fft.rfft(X, axis=1)
    p = (F.real ** 2 + F.imag ** 2)
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0)
    return freqs, p, F


def dominant_freq_idx(p):
    """index of max power excluding DC (bin 0)."""
    pp = p.copy()
    pp[:, 0] = -1.0
    return np.argmax(pp, axis=1)


def spectral_centroid(freqs, p):
    pp = p.copy()
    pp[:, 0] = 0.0  # drop DC
    num = (pp * freqs[None, :]).sum(axis=1)
    den = pp.sum(axis=1) + 1e-12
    return num / den


def spectral_entropy(p):
    pp = p.copy()
    pp[:, 0] = 0.0
    s = pp.sum(axis=1, keepdims=True) + 1e-12
    q = pp / s
    q = np.clip(q, 1e-12, 1.0)
    ent = -(q * np.log(q)).sum(axis=1)
    return ent


def spectral_spread(freqs, p, cen):
    pp = p.copy()
    pp[:, 0] = 0.0
    den = pp.sum(axis=1) + 1e-12
    var = (pp * (freqs[None, :] - cen[:, None]) ** 2).sum(axis=1) / den
    return np.sqrt(var)


def phase_at(F, k):
    """phase of FFT bin k (radians)."""
    return np.angle(F[:, k])


def sign_changes(X):
    s = np.sign(X)
    s[s == 0] = 1
    return (np.abs(np.diff(s, axis=1)) > 0).sum(axis=1)


def mean_run_length(X):
    n = X.shape[1]
    sc = sign_changes(X)
    return n / (sc + 1.0)


def power_at_period(freqs, p, period):
    """interpolate power at frequency 1/period."""
    f0 = 1.0 / period
    out = np.zeros(p.shape[0])
    for i in range(1, len(freqs)):
        if freqs[i - 1] <= f0 <= freqs[i]:
            w = (f0 - freqs[i - 1]) / (freqs[i] - freqs[i - 1] + 1e-12)
            out = p[:, i - 1] * (1 - w) + p[:, i] * w
            return out
    # exact-bin fallback: nearest
    idx = np.argmin(np.abs(freqs - f0))
    return p[:, idx]


def best_period_grid(Xc, periods):
    """for each row, grid-search sinusoid period maximizing fit (via projection
    onto sin/cos of that period); return best period and its normalized power."""
    n, L = Xc.shape
    t = np.arange(L)
    best_p = np.zeros(n)
    best_pow = np.zeros(n)
    tot = (Xc ** 2).sum(axis=1) + 1e-12
    for P in periods:
        w = 2 * np.pi / P
        cs = np.cos(w * t)
        sn = np.sin(w * t)
        a = Xc @ cs
        b = Xc @ sn
        # power captured (normalized basis)
        pw = (a ** 2) / (cs @ cs + 1e-12) + (b ** 2) / (sn @ sn + 1e-12)
        upd = pw > best_pow
        best_pow[upd] = pw[upd]
        best_p[upd] = P
    return best_p, best_pow / tot


# ================= build candidates =================
cands = {}

for name, Xc, Xraw in [("d", Dc, Draw), ("w", Wc, Wraw)]:
    freqs, p, F = rfft_power(Xc)
    L = Xc.shape[1]

    # dominant frequency (index -> frequency value)
    di = dominant_freq_idx(p)
    cands[f"{name}_domfreq"] = freqs[di]
    cands[f"{name}_domperiod"] = clean(np.where(di > 0, L / np.maximum(di, 1), 0.0))
    # power fraction in the dominant bin
    ptot = p[:, 1:].sum(axis=1) + 1e-12
    cands[f"{name}_dompow_frac"] = p[np.arange(len(di)), di] / ptot

    # spectral centroid / spread / entropy
    cen = spectral_centroid(freqs, p)
    cands[f"{name}_centroid"] = cen
    cands[f"{name}_spread"] = spectral_spread(freqs, p, cen)
    cands[f"{name}_entropy"] = spectral_entropy(p)

    # phase of each low-frequency bin (k=1..min(4,F-1)); use cos & sin comps
    for k in range(1, min(5, p.shape[1])):
        ph = phase_at(F, k)
        cands[f"{name}_phase{k}_cos"] = np.cos(ph)
        cands[f"{name}_phase{k}_sin"] = np.sin(ph)
        cands[f"{name}_phase{k}_raw"] = ph

    # sign structure of centered series
    cands[f"{name}_signchanges"] = sign_changes(Xc).astype(float)
    cands[f"{name}_meanrun"] = mean_run_length(Xc)

    # band power ratios: period-3 vs period-4 vs period-2
    p2 = power_at_period(freqs, p, 2.0)
    p3 = power_at_period(freqs, p, 3.0)
    p4 = power_at_period(freqs, p, 4.0)
    p6 = power_at_period(freqs, p, 6.0)
    cands[f"{name}_p3_over_p4"] = clean(p3 / (p4 + 1e-9))
    cands[f"{name}_p3_over_p2"] = clean(p3 / (p2 + 1e-9))
    cands[f"{name}_p3_frac"] = clean(p3 / ptot)
    cands[f"{name}_p4_frac"] = clean(p4 / ptot)
    cands[f"{name}_p2_frac"] = clean(p2 / ptot)
    cands[f"{name}_p6_frac"] = clean(p6 / ptot)
    cands[f"{name}_lowhigh"] = clean((p2 + p3 + p4) / (ptot + 1e-9))

    # best-fit sinusoid period grid search
    if name == "d":
        periods = np.arange(2.0, 9.0, 0.25)
    else:
        periods = np.arange(2.0, 7.0, 0.25)
    bp, bpn = best_period_grid(Xc, periods)
    cands[f"{name}_bestperiod"] = bp
    cands[f"{name}_bestperiod_powfrac"] = clean(bpn)

    # high-frequency energy fraction (fast oscillation)
    hf = p[:, max(1, p.shape[1] // 2):].sum(axis=1)
    cands[f"{name}_hf_frac"] = clean(hf / ptot)

# ---------- cross daily/weekly period agreement ----------
cands["dw_domperiod_diff"] = clean(cands["d_domperiod"] - cands["w_domperiod"])
cands["dw_centroid_prod"] = clean(cands["d_centroid"] * cands["w_centroid"])

# ================= score =================
rows = []
for k, v in cands.items():
    v = clean(v)
    rows.append((k, spear(v), partial(v)))

rows.sort(key=lambda r: -abs(r[2]))

print(f"{'name':32s} {'spearman':>9s} {'partial':>9s}")
print("-" * 54)
for nm, sp, pt in rows:
    flag = "  <== INTERESTING" if abs(pt) >= 0.04 else ""
    print(f"{nm:32s} {sp:>9.4f} {pt:>9.4f}{flag}")

out = {
    "candidates": [{"name": nm, "spearman": sp, "partial": pt} for nm, sp, pt in rows],
    "no_kaggle_submission_made": True,
}
with open(f"{ROOT}/outputs/disc_E.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote outputs/disc_E.json")
