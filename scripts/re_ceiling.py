"""Ceiling test: can a from-scratch CV model beat v6? And what's left in the residual?"""
import numpy as np, pandas as pd, scipy.stats as ss
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT="D:/github/datathon2026_playground"
df=pd.read_csv(f"{ROOT}/kaggle/train.csv")
d=np.load(f"{ROOT}/outputs/v6_oof.npz"); oof,y=d['oof_rank'].astype(float),d['y'].astype(float)
n=len(y); ry=ss.rankdata(y); roof=ss.rankdata(oof)
def sp(a,b): return ss.spearmanr(a,b).correlation
LAT=['skor_motivasi','skor_kedisiplinan','skor_literasi','skor_minat_belajar',
     'indeks_kehadiran','skor_ekstrakurikuler','jarak_rumah_km','jumlah_saudara']
WEEK=[f'nilai_minggu_{i:02d}' for i in range(1,13)]; DAY=[f'aktivitas_hari_{i:02d}' for i in range(1,17)]
W=df[WEEK].values.astype(float); A=df[DAY].values.astype(float)
Wc=W-W.mean(1,keepdims=True); Ac=A-A.mean(1,keepdims=True)

# rich engineered matrix (the good structural features)
C={}
for c in LAT: C[c]=df[c].values.astype(float)
Z={c:df[c].values-df[c].values.mean() for c in LAT}
C['motiv_kedis']=Z['skor_motivasi']*Z['skor_kedisiplinan']
C['week_mean']=W.mean(1); C['week_std']=W.std(1)
C['week_lag1ac']=np.array([np.corrcoef(r[:-1],r[1:])[0,1] if r[:-1].std()>0 else 0 for r in Wc])
C['day_lag1ac']=np.array([np.corrcoef(r[:-1],r[1:])[0,1] for r in Ac])
C['day_lag3ac']=np.array([np.corrcoef(r[:-3],r[3:])[0,1] for r in Ac])
C['completion']=(df.tugas_selesai/np.maximum(df.tugas_diberikan,1)).values
C['tryout']=df.skor_tryout.values.astype(float)
# throw in ALL pairwise latent products too
for a,b in combinations(LAT,2): C[f'{a}*{b}']=Z[a]*Z[b]
names=list(C); X=np.c_[[C[k] for k in names]].T

# OOF CV regressor to y (proper, no leakage)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
myoof=np.zeros(n)
for tr,va in skf.split(X,y):
    m=HistGradientBoostingRegressor(max_depth=4,max_iter=400,learning_rate=0.05,
        l2_regularization=1.0,random_state=0).fit(X[tr],y[tr])
    myoof[va]=m.predict(X[va])
print("From-scratch rich-feature OOF spearman vs y:", round(sp(myoof,y),4))
print("v6 OOF spearman vs y:                       ", round(sp(oof,y),4))
# does blending my model into v6 help rank?
for w in [0.0,0.2,0.4,0.5]:
    bl=(1-w)*roof+w*ss.rankdata(myoof)
    print(f"  blend v6*(1-{w})+mine*{w}: sp={sp(bl,y):.4f}")

# The residual that v6 misses — does my independent model explain it?
M=np.c_[np.ones(n),roof]; resid=ry-M@np.linalg.lstsq(M,ry,rcond=None)[0]
print("\nspearman(my_oof, v6-residual):", round(sp(myoof,resid),4),
      " (near 0 => nothing left to grab)")

# How much of v6's own rank is 'unexplained noise' vs recoverable structure?
# Correlate two independent halves? Instead: agreement of my model with v6
print("spearman(my_oof, v6):", round(sp(myoof,oof),4))
