"""Discovery E final refinement: confirm daily-PERIOD signal via INDEPENDENT
period estimators (not just FFT projection), and probe period-drift.

Established: d_bestperiod_fine (FFT/projection best period) partial -0.0557.
Cross-check with autocorrelation-based period, cycle counting, and half-split
drift to confirm this is genuine planted structure orthogonal to v6.
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
Draw = tr[DAY].to_numpy(float)
Dc = Draw - Draw.mean(axis=1, keepdims=True)

cands = {}

def best_period_grid(Xc, grid):
    n, L = Xc.shape; t = np.arange(L)
    best_p = np.zeros(n); best_pow = np.full(n, -1.0)
    for P in grid:
        w = 2*np.pi/P; cs = np.cos(w*t); sn = np.sin(w*t)
        a = Xc @ cs; b = Xc @ sn
        pw = (a**2)/(cs@cs+1e-12) + (b**2)/(sn@sn+1e-12)
        upd = pw > best_pow; best_pow[upd]=pw[upd]; best_p[upd]=P
    return best_p

# 1. Autocorrelation-based period: lag of first local max of biased autocorr (lags>=1)
def autocorr_period(Xc, max_lag):
    n, L = Xc.shape
    acs = np.zeros((n, max_lag+1))
    denom = (Xc**2).sum(axis=1) + 1e-12
    for lag in range(1, max_lag+1):
        acs[:, lag] = (Xc[:, :L-lag]*Xc[:, lag:]).sum(axis=1)/denom
    # first local maximum among lags 1..max_lag-1
    period = np.zeros(n)
    for lag in range(2, max_lag):
        is_peak = (acs[:,lag] > acs[:,lag-1]) & (acs[:,lag] >= acs[:,lag+1]) & (period==0)
        period[is_peak] = lag
    period[period==0] = max_lag  # no peak found
    return period, acs
ac_per, acs = autocorr_period(Dc, 8)
cands["d_autocorr_period"] = ac_per
# lag of minimum autocorr (anti-phase) — half period marker
cands["d_min_ac_lag"] = np.argmin(acs[:,1:], axis=1).astype(float) + 1

# 2. Cycle count = sign_changes/2 ; effective period = 2L/signchanges
s = np.sign(Dc); s[s==0]=1
sc = (np.abs(np.diff(s,axis=1))>0).sum(axis=1).astype(float)
cands["d_eff_period_signs"] = clean(2*Dc.shape[1]/(sc+1e-9))
cands["d_cycles"] = sc/2.0

# 3. Period drift: best period on first half vs second half
half = Dc.shape[1]//2
bp1 = best_period_grid(Dc[:,:half], np.arange(2.0,7.01,0.1))
bp2 = best_period_grid(Dc[:,half:], np.arange(2.0,7.01,0.1))
cands["d_period_drift"] = clean(bp2 - bp1)
cands["d_period_half_mean"] = clean(0.5*(bp1+bp2))

# 4. reference full-series fine period (should reproduce -0.0557)
cands["d_bestperiod_fine"] = best_period_grid(Dc, np.arange(2.0,8.01,0.05))

# 5. Consensus: average of z-scored period estimators (FFT + autocorr + sign)
def z(a):
    a = clean(a); return (a-a.mean())/(a.std()+1e-12)
consensus = z(cands["d_bestperiod_fine"]) + z(ac_per) + z(cands["d_eff_period_signs"])
cands["d_period_consensus"] = consensus

rows = []
for k,v in cands.items():
    v = clean(v); rows.append((k, spear(v), partial(v)))
rows.sort(key=lambda r: -abs(r[2]))
print(f"{'name':26s} {'spearman':>9s} {'partial':>9s}")
print("-"*50)
for nm,sp,pt in rows:
    flag = "  <== INTERESTING" if abs(pt)>=0.04 else ""
    print(f"{nm:26s} {sp:>9.4f} {pt:>9.4f}{flag}")

out = {"candidates":[{"name":nm,"spearman":sp,"partial":pt} for nm,sp,pt in rows],
       "no_kaggle_submission_made": True}
with open(f"{ROOT}/outputs/disc_E3.json","w") as f: json.dump(out,f,indent=2)
print("\nwrote outputs/disc_E3.json")
