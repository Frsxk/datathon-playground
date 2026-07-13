"""Test cleaner AR-parameter extractions as candidate features to add to v6."""
import numpy as np, pandas as pd, scipy.stats as ss
ROOT="D:/github/datathon2026_playground"
df=pd.read_csv(f"{ROOT}/kaggle/train.csv")
d=np.load(f"{ROOT}/outputs/v6_oof.npz"); oof,y=d['oof_rank'].astype(float),d['y'].astype(float)
n=len(y); ry=ss.rankdata(y); roof=ss.rankdata(oof)
WEEK=[f'nilai_minggu_{i:02d}' for i in range(1,13)]; DAY=[f'aktivitas_hari_{i:02d}' for i in range(1,17)]
W=df[WEEK].values.astype(float); A=df[DAY].values.astype(float)
Wc=W-W.mean(1,keepdims=True); Ac=A-A.mean(1,keepdims=True)
def sp(a,b): return ss.spearmanr(a,b).correlation
def partial_v6(f):
    fr=ss.rankdata(f); M=np.c_[np.ones(n),roof]; r=lambda v:v-M@np.linalg.lstsq(M,v,rcond=None)[0]
    return np.corrcoef(r(fr),r(ry))[0,1]

def ar1(series):
    out=np.zeros(len(series))
    for i,r in enumerate(series):
        x0,x1=r[:-1],r[1:]
        out[i]=np.polyfit(x0,x1,1)[0] if x0.std()>0 else 0
    return out

feats={
 'week_ar1_coef': ar1(Wc),
 'week_lag1ac': np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Wc]),
 'week_std': W.std(1),
 'week_ar2_coef': np.array([np.polyfit(np.c_[r[1:-1],r[:-2]],r[2:],1)[0][0] if r[:-2].std()>0 else 0 for r in Wc]) if False else np.zeros(n),
 # daily: v6 uses period-3; test lag2 ac and the AR sign-flip pattern
 'day_lag2ac': np.array([np.corrcoef(r[:-2],r[2:])[0,1] for r in Ac]),
 'day_lag3ac': np.array([np.corrcoef(r[:-3],r[3:])[0,1] for r in Ac]),
 'day_ar1_coef': ar1(Ac),
 # combined AR persistence proxy
 'week_ac_avg': np.array([np.mean([np.corrcoef(r[:-k],r[k:])[0,1] for k in (1,2)]) for r in Wc]),
}
print(f"{'feature':22s}{'sp_vs_y':>10s}{'partial_v6':>12s}")
for k,v in feats.items():
    if np.std(v)==0: continue
    print(f"{k:22s}{sp(v,y):10.3f}{partial_v6(v):12.3f}")
