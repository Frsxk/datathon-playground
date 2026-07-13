"""Residual attack + generative-form probes. Single quick model fits only."""
import numpy as np, pandas as pd, scipy.stats as ss
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT = "D:/github/datathon2026_playground"
df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
oof, y = d['oof_rank'].astype(float), d['y'].astype(float)
ry = ss.rankdata(y); roof = ss.rankdata(oof); n=len(y)

LAT = ['skor_motivasi','skor_kedisiplinan','skor_literasi','skor_minat_belajar',
       'indeks_kehadiran','skor_ekstrakurikuler','jarak_rumah_km','jumlah_saudara']
WEEK = [f'nilai_minggu_{i:02d}' for i in range(1,13)]
DAY  = [f'aktivitas_hari_{i:02d}' for i in range(1,17)]
def sp(a,b): return ss.spearmanr(a,b).correlation
def partial_v6(feat):
    fr=ss.rankdata(feat); A=np.c_[np.ones(n),roof]
    r=lambda v: v-A@np.linalg.lstsq(A,v,rcond=None)[0]
    return np.corrcoef(r(fr),r(ry))[0,1]

# rich matrix
cols={}
for c in LAT: cols[c]=df[c].values.astype(float)
Z={c:(df[c].values-df[c].values.mean()) for c in LAT}
for a,b in combinations(LAT,2): cols[f'{a}*{b}']=Z[a]*Z[b]
W=df[WEEK].values.astype(float); A=df[DAY].values.astype(float)
Wc=W-W.mean(1,keepdims=True); Ac=A-A.mean(1,keepdims=True)
cols['week_mean']=W.mean(1); cols['week_std']=W.std(1)
cols['week_slope']=np.polyfit(np.arange(12),W.T,1)[0]
cols['week_lag1ac']=np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Wc])
cols['day_std']=A.std(1)
cols['day_lag1ac']=np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Ac])
cols['day_lag3ac']=np.array([np.corrcoef(r[:-3],r[3:])[0,1] if r[:-3].std()>0 else 0 for r in Ac])
cols['completion']=df['tugas_selesai'].values/np.maximum(df['tugas_diberikan'].values,1)
cols['skor_tryout']=df['skor_tryout'].values.astype(float)
names=list(cols); X=np.c_[[cols[k] for k in names]].T

# ---------- Analysis 2: residual attack ----------
Ar=np.c_[np.ones(n),roof]; resid=ry-Ar@np.linalg.lstsq(Ar,ry,rcond=None)[0]
m=HistGradientBoostingRegressor(max_depth=3,max_iter=300,learning_rate=0.05,random_state=0)
m.fit(X,resid)
pr=m.predict(X)
print("residual model in-sample spearman(pred,resid):", round(sp(pr,resid),3))
pi=permutation_importance(m,X,resid,n_repeats=5,random_state=0,scoring='r2')
order=np.argsort(-pi.importances_mean)
print("\nTop permutation importances predicting the y-vs-v6 RESIDUAL:")
for i in order[:15]:
    print(f"  {names[i]:34s}{pi.importances_mean[i]:8.4f}  (marg sp {sp(X[:,i],resid):+.3f})")

# ---------- Analysis 4: ceiling check ----------
print("\n"+"="*60,"\nANALYSIS 4: how much does week_mean + known structs explain?")
inter=Z['skor_motivasi']*Z['skor_kedisiplinan']
known=np.c_[np.ones(n),ss.rankdata(cols['week_mean']),ss.rankdata(inter),
            ss.rankdata(cols['day_lag3ac']),ss.rankdata(cols['week_lag1ac'])]
b=np.linalg.lstsq(known,ry,rcond=None)[0]; predk=known@b
print("  spearman(week_mean+inter+lag structs, y):", round(sp(predk,y),3))
print("  vs v6 spearman:", round(sp(oof,y),3))

# ---------- Analysis 3b: probe generative forms ----------
print("\n"+"="*60,"\nGenerative-form probes (spearman vs y | partial vs v6):")
zc={c:(df[c].values-df[c].values.mean())/df[c].values.std() for c in LAT}
probes={
 'week_std':cols['week_std'],
 'week_std*week_mean':ss.rankdata(cols['week_std'])+ss.rankdata(cols['week_mean']),
 'week_range':W.max(1)-W.min(1),
 'week_last_minus_first':W[:,-1]-W[:,0],
 'day_amplitude':A.max(1)-A.min(1),
 'sum_all_lat':sum(zc[c] for c in LAT),
 'max_lat':np.max(np.c_[[zc[c] for c in LAT]].T,1),
 'min_lat':np.min(np.c_[[zc[c] for c in LAT]].T,1),
 'motiv*kedis':inter,
 'tryout':cols['skor_tryout'],
 'week_mean':cols['week_mean'],
}
for k,v in probes.items():
    print(f"  {k:26s} sp={sp(v,y):+.3f}  partial_v6={partial_v6(v):+.3f}")
