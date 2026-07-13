"""Dissect the weekly series generative form: why does week_std predict class?"""
import numpy as np, pandas as pd, scipy.stats as ss

ROOT="D:/github/datathon2026_playground"
df=pd.read_csv(f"{ROOT}/kaggle/train.csv")
d=np.load(f"{ROOT}/outputs/v6_oof.npz"); oof,y=d['oof_rank'].astype(float),d['y'].astype(float)
ry=ss.rankdata(y); roof=ss.rankdata(oof); n=len(y)
WEEK=[f'nilai_minggu_{i:02d}' for i in range(1,13)]
DAY=[f'aktivitas_hari_{i:02d}' for i in range(1,17)]
W=df[WEEK].values.astype(float); A=df[DAY].values.astype(float)
def sp(a,b): return ss.spearmanr(a,b).correlation
def partial_v6(f):
    fr=ss.rankdata(f); M=np.c_[np.ones(n),roof]; r=lambda v:v-M@np.linalg.lstsq(M,v,rcond=None)[0]
    return np.corrcoef(r(fr),r(ry))[0,1]

print("=== Weekly series stats by class ===")
Wc=W-W.mean(1,keepdims=True)
for lab in [0,1,2,3]:
    m=y==lab; sub=W[m]
    ac1=np.mean([np.corrcoef(r[:-1],r[1:])[0,1] for r in (sub-sub.mean(1,keepdims=True))])
    print(f" cls{lab}: overall_mean={sub.mean():.2f} within_std={sub.std(1).mean():.3f} lag1ac={ac1:.3f} betwmean_std={sub.mean(1).std():.3f}")

print("\n=== Is weekly an AR(1)? Fit x_t = a*x_{t-1}+e per class (pooled) ===")
for lab in [0,1,2,3]:
    m=y==lab; sub=Wc[m]
    x0=sub[:,:-1].ravel(); x1=sub[:,1:].ravel()
    a=np.polyfit(x0,x1,1)[0]; res=x1-a*x0
    print(f" cls{lab}: AR1 coef={a:.3f} innov_std={res.std():.3f} within_std={sub.std(1).mean():.3f}")

# Decompose week_std: is it driven by innovation variance or by AR persistence?
print("\n=== week_std partial after controlling for week_lag1ac ===")
ac1=np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Wc])
wstd=W.std(1)
# partial of wstd vs y controlling BOTH v6 and lag1ac
def partial_ctrl(f,ctrls):
    fr=ss.rankdata(f); M=np.c_[np.ones(n)]+0
    M=np.c_[np.ones(n)]+0
    M=np.column_stack([np.ones(n)]+[ss.rankdata(c) for c in ctrls])
    r=lambda v:v-M@np.linalg.lstsq(M,v,rcond=None)[0]
    return np.corrcoef(r(fr),r(ry))[0,1]
print(" week_std partial vs y | (v6):", round(partial_v6(wstd),3))
print(" week_std partial vs y | (lag1ac):", round(partial_ctrl(wstd,[ac1]),3))
print(" week_std partial vs y | (v6,lag1ac):", round(partial_ctrl(wstd,[roof,ac1]),3))
print(" lag1ac  partial vs y | (v6,week_std):", round(partial_ctrl(ac1,[roof,wstd]),3))

# Daily amplitude / period probe (v6 uses period-3). Check daily variance driver too
print("\n=== Daily series: innovation vs oscillation by class ===")
Ac=A-A.mean(1,keepdims=True)
for lab in [0,1,2,3]:
    m=y==lab; sub=Ac[m]
    ac1=np.mean([np.corrcoef(r[:-1],r[1:])[0,1] for r in sub])
    ac3=np.mean([np.corrcoef(r[:-3],r[3:])[0,1] for r in sub])
    print(f" cls{lab}: within_std={sub.std(1).mean():.3f} lag1ac={ac1:.3f} lag3ac={ac3:.3f}")

# Cross: does a *daily-weekly* or activity*grade interaction carry residual signal?
print("\n=== Cross-block probes (partial vs v6) ===")
probes={
 'week_mean * day_mean': ss.rankdata(W.mean(1))+ss.rankdata(A.mean(1)),
 'week_std * day_lag3ac': W.std(1)*np.array([np.corrcoef(r[:-3],r[3:])[0,1] for r in Ac]),
 'completion * week_mean': (df.tugas_selesai/np.maximum(df.tugas_diberikan,1)).values*W.mean(1),
 'tryout * week_mean': df.skor_tryout.values*W.mean(1),
 'week_ar1_innov_std': np.array([ (r[1:]-np.polyfit(r[:-1],r[1:],1)[0]*r[:-1]).std() if r[:-1].std()>0 else 0 for r in Wc]),
}
for k,v in probes.items():
    print(f"  {k:26s} sp={sp(v,y):+.3f} partial_v6={partial_v6(v):+.3f}")
