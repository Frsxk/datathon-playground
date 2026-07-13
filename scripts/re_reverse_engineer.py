"""Reverse-engineer the hidden performance score. Light, single-fit models only."""
import numpy as np, pandas as pd, scipy.stats as ss
from itertools import combinations

ROOT = "D:/github/datathon2026_playground"
df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
oof, y = d['oof_rank'].astype(float), d['y'].astype(float)
ry = ss.rankdata(y)
roof = ss.rankdata(oof)
n = len(y)

LAT = ['skor_motivasi','skor_kedisiplinan','skor_literasi','skor_minat_belajar',
       'indeks_kehadiran','skor_ekstrakurikuler','jarak_rumah_km','jumlah_saudara']
WEEK = [f'nilai_minggu_{i:02d}' for i in range(1,13)]
DAY  = [f'aktivitas_hari_{i:02d}' for i in range(1,17)]

def sp(a,b): return ss.spearmanr(a,b).correlation

def partial_vs_v6(feat):
    """spearman partial corr of feat with y, controlling for oof (rank residuals)."""
    fr = ss.rankdata(feat)
    # residualize fr and ry on roof (linear on ranks)
    def resid(v):
        A = np.c_[np.ones(n), roof]
        beta = np.linalg.lstsq(A, v, rcond=None)[0]
        return v - A@beta
    return np.corrcoef(resid(fr), resid(ry))[0,1]

# ---------- build rich feature set ----------
feats = {}
for c in LAT: feats[c] = df[c].values.astype(float)
for c in LAT: feats[c+'^2'] = df[c].values.astype(float)**2
# pairwise products of latents (centered)
Z = {c: (df[c].values - df[c].values.mean()) for c in LAT}
for a,b in combinations(LAT,2):
    feats[f'{a}*{b}'] = Z[a]*Z[b]
# weekly summaries
W = df[WEEK].values.astype(float)
feats['week_mean'] = W.mean(1)
feats['week_std']  = W.std(1)
xw = np.arange(12)
feats['week_slope'] = np.polyfit(xw, W.T, 1)[0]
Wc = W - W.mean(1,keepdims=True)
feats['week_lag1ac'] = np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Wc])
# daily summaries
A = df[DAY].values.astype(float)
feats['day_mean'] = A.mean(1)
feats['day_std']  = A.std(1)
Ac = A - A.mean(1,keepdims=True)
feats['day_lag1ac'] = np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Ac])
feats['day_lag3ac'] = np.array([np.corrcoef(r[:-3],r[3:])[0,1] if r[:-3].std()>0 else 0 for r in Ac])
# completion, extras
feats['completion'] = df['tugas_selesai'].values / np.maximum(df['tugas_diberikan'].values,1)
feats['skor_tryout'] = df['skor_tryout'].values.astype(float)

print("="*70)
print("MARGINAL spearman vs y, and PARTIAL vs v6 (|partial|>=0.05 interesting)")
print("="*70)
rows=[]
for name,v in feats.items():
    if np.std(v)==0: continue
    rows.append((name, sp(v,y), partial_vs_v6(v)))
rows.sort(key=lambda r:-abs(r[2]))
print(f"{'feature':28s}{'sp_vs_y':>10s}{'partial_v6':>12s}")
for name,s,p in rows[:30]:
    flag = ' <<<' if abs(p)>=0.05 else ''
    print(f"{name:28s}{s:10.3f}{p:12.3f}{flag}")

# ---------- Analysis 3: linear reg of oof on 8 latents (which weights nonzero) ----------
print("\n"+"="*70)
print("ANALYSIS 3: linear reg of v6 oof_rank on 8 latents (standardized)")
print("="*70)
Xl = np.c_[[ (df[c].values-df[c].values.mean())/df[c].values.std() for c in LAT]].T
Xl = np.c_[np.ones(n), Xl]
beta = np.linalg.lstsq(Xl, roof, rcond=None)[0]
for c,b in zip(['intercept']+LAT, beta):
    print(f"  {c:24s}{b:10.2f}")
pred = Xl@beta
print("  R2 of latents->oof_rank:", 1-np.var(roof-pred)/np.var(roof))

# z-sum of latents vs target
print("\nz-sum of all 8 latents (signed by reg) vs y spearman:")
signs = np.sign(beta[1:])
zsum = np.c_[[ signs[i]*(df[c].values-df[c].values.mean())/df[c].values.std() for i,c in enumerate(LAT)]].T.sum(1)
print("  ", sp(zsum, y), " partial vs v6:", partial_vs_v6(zsum))
