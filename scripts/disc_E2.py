"""Discovery E refinement: sharpen the daily best-fit PERIOD signal.

disc_E.py found d_bestperiod (grid-searched dominant sinusoid period of the
time-centered daily series) has partial -0.0545, orthogonal to v6. Here we
refine that direction: finer period grid, parabolic peak-period interpolation
on the FFT (sub-bin resolution), robustness variants, and interactions.
All target-free.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"

d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float); y = d["y"].astype(float)
ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur); tvar = np.sqrt((ty**2).mean())

def spear(v):
    v = np.asarray(v, float)
    if np.std(v) < 1e-12: return 0.0
    return float(np.corrcoef(rankdata(v), ry)[0, 1])

def partial(v):
    v = np.asarray(v, float)
    if np.std(v) < 1e-12: return 0.0
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur); fv = np.sqrt((fr**2).mean())
    return float((fr*ty).mean()/(fv*tvar+1e-12)) if fv > 1e-9 else 0.0

def clean(a):
    a = np.asarray(a, float); a[~np.isfinite(a)] = 0.0; return a

tr = pd.read_csv(f"{ROOT}/kaggle/train.csv")
DAY = sorted([c for c in tr.columns if c.startswith("aktivitas_hari_")])
WK  = sorted([c for c in tr.columns if c.startswith("nilai_minggu_")])
Draw = tr[DAY].to_numpy(float); Wraw = tr[WK].to_numpy(float)
Dc = Draw - Draw.mean(axis=1, keepdims=True)
Wc = Wraw - Wraw.mean(axis=1, keepdims=True)

cands = {}

def best_period_grid(Xc, periods, return_r2=False):
    n, L = Xc.shape; t = np.arange(L)
    best_p = np.zeros(n); best_pow = np.full(n, -1.0)
    tot = (Xc**2).sum(axis=1) + 1e-12
    for P in periods:
        w = 2*np.pi/P; cs = np.cos(w*t); sn = np.sin(w*t)
        a = Xc @ cs; b = Xc @ sn
        pw = (a**2)/(cs@cs+1e-12) + (b**2)/(sn@sn+1e-12)
        upd = pw > best_pow
        best_pow[upd] = pw[upd]; best_p[upd] = P
    if return_r2:
        return best_p, best_pow/tot
    return best_p

# 1. Fine grid best-fit period for daily and weekly
for name, Xc in [("d", Dc), ("w", Wc)]:
    grid = np.arange(2.0, 8.01, 0.05) if name=="d" else np.arange(2.0, 6.01, 0.05)
    bp, r2 = best_period_grid(Xc, grid, return_r2=True)
    cands[f"{name}_bestperiod_fine"] = bp
    cands[f"{name}_bestperiod_r2"] = r2
    # inverse (best frequency)
    cands[f"{name}_bestfreq_fine"] = 1.0/np.maximum(bp, 1e-6)

# 2. Parabolic sub-bin peak period from FFT (log-magnitude parabola around argmax)
def parabolic_peak_period(Xc):
    F = np.fft.rfft(Xc, axis=1)
    p = F.real**2 + F.imag**2
    L = Xc.shape[1]; nb = p.shape[1]
    pp = p.copy(); pp[:,0] = -1
    k = np.argmax(pp, axis=1)
    n = Xc.shape[0]
    kf = k.astype(float)
    # parabolic interpolation using neighbors (guard edges)
    lg = np.log(p + 1e-12)
    delta = np.zeros(n)
    valid = (k >= 1) & (k <= nb-2)
    km = np.clip(k-1, 0, nb-1); kp = np.clip(k+1, 0, nb-1)
    a = lg[np.arange(n), km]; b = lg[np.arange(n), k]; c = lg[np.arange(n), kp]
    denom = (a - 2*b + c)
    dd = 0.5*(a - c)/np.where(np.abs(denom)<1e-9, 1e-9, denom)
    delta = np.where(valid, dd, 0.0)
    kf = kf + np.clip(delta, -0.5, 0.5)
    period = L/np.maximum(kf, 1e-6)
    return period, kf

for name, Xc in [("d", Dc), ("w", Wc)]:
    per, kf = parabolic_peak_period(Xc)
    cands[f"{name}_parab_period"] = clean(per)
    cands[f"{name}_parab_binfreq"] = clean(kf)

# 3. Period bucketed / distance from period-3 (the known daily oscillation)
bp_d = cands["d_bestperiod_fine"]
cands["d_dist_from_p3"] = np.abs(bp_d - 3.0)
cands["d_dist_from_p3_signed"] = bp_d - 3.0
cands["d_period_below3"] = (bp_d < 3.0).astype(float)
cands["d_period_gt4"] = (bp_d > 4.0).astype(float)

# 4. Ratio of energy captured by period-3 fit vs best period fit
def fit_r2_at(Xc, P):
    L = Xc.shape[1]; t = np.arange(L); w = 2*np.pi/P
    cs = np.cos(w*t); sn = np.sin(w*t)
    a = Xc @ cs; b = Xc @ sn
    pw = (a**2)/(cs@cs+1e-12) + (b**2)/(sn@sn+1e-12)
    return pw/((Xc**2).sum(axis=1)+1e-12)
r2_p3 = fit_r2_at(Dc, 3.0)
cands["d_r2p3_over_best"] = clean(r2_p3 / (cands["d_bestperiod_r2"]+1e-9))
cands["d_r2_p3"] = r2_p3

# 5. Weighted mean period across spectrum (period-domain centroid)
def period_centroid(Xc):
    F = np.fft.rfft(Xc, axis=1); p = F.real**2 + F.imag**2
    L = Xc.shape[1]; nb = p.shape[1]
    ks = np.arange(nb, dtype=float); periods = np.where(ks>0, L/np.maximum(ks,1e-6), 0.0)
    pp = p.copy(); pp[:,0]=0.0
    den = pp.sum(axis=1)+1e-12
    return (pp*periods[None,:]).sum(axis=1)/den
cands["d_period_centroid"] = period_centroid(Dc)
cands["w_period_centroid"] = period_centroid(Wc)

# 6. Interactions of daily best-period with known captured pieces? Keep pure.
#    Instead: daily period * daily oscillation amplitude (r2) — nonlinear combo
cands["d_period_x_r2"] = clean(bp_d * cands["d_bestperiod_r2"])

# score
rows = []
for k,v in cands.items():
    v = clean(v); rows.append((k, spear(v), partial(v)))
rows.sort(key=lambda r: -abs(r[2]))
print(f"{'name':28s} {'spearman':>9s} {'partial':>9s}")
print("-"*52)
for nm,sp,pt in rows:
    flag = "  <== INTERESTING" if abs(pt)>=0.04 else ""
    print(f"{nm:28s} {sp:>9.4f} {pt:>9.4f}{flag}")

out = {"candidates":[{"name":nm,"spearman":sp,"partial":pt} for nm,sp,pt in rows],
       "no_kaggle_submission_made": True}
with open(f"{ROOT}/outputs/disc_E2.json","w") as f: json.dump(out,f,indent=2)
print("\nwrote outputs/disc_E2.json")
