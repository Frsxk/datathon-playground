#!/usr/bin/env python3
"""Signal discovery: screen many NEW feature families for signal the current
52-feature set misses. Sensitive screening (accuracy quantizes small gains):
  1. Spearman(feat, target)              -- raw monotone signal
  2. Partial Spearman | current OOF rank -- signal ORTHOGONAL to what we have
  3. Ridge/logistic linear-probe rank    -- oblique direction trees can't see

CV-safe: the 'current rank' is the ordinal OOF score. No Kaggle submission.
Writes outputs/exp_signal_screen.json.
"""
from __future__ import annotations
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from cv_harness import build_features, quartile_bin, z, SEEDS

tr = pd.read_csv("kaggle/train.csv"); y = tr["target"]; yv = y.values
_f = build_features(tr, "full")
Xcur = _f.drop(columns=[c for c in _f.columns if c.startswith("kelas")])
W = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
Dc = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
Wm = tr[W].to_numpy(float); Dm = tr[Dc].to_numpy(float)
print(f"current nfeat={Xcur.shape[1]}")

# ---- current latent rank (ordinal OOF, seed 42) as the 'already-captured' signal
def ordinal_oof(X, seed=42):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed); s = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr_i], X.iloc[va_i], y.iloc[tr_i]
        for k in range(3):
            b = XGBClassifier(n_estimators=550, max_depth=4, learning_rate=0.04, subsample=0.9,
                colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5, random_state=seed,
                n_jobs=-1, eval_metric="logloss", verbosity=0)
            b.fit(Xtr, (ytr > k).astype(int)); s[va_i] += b.predict_proba(Xva)[:, 1]
    return s
cur_rank = rankdata(ordinal_oof(Xcur))
print("built current ordinal rank")

def longest_run(mask):  # per-row longest True run
    out = np.zeros(len(mask))
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        out[i] = best
    return out

# ---------------- NEW candidate feature bank ----------------
cand = {}
# (A) temporal PATTERN of weekly grade-changes (beyond magnitude)
d = np.diff(Wm, axis=1)
cand["wk_sign_changes"] = (np.diff(np.sign(Wm), axis=1) != 0).sum(1)
cand["wk_longest_pos_run"] = longest_run(Wm > 0)
cand["wk_longest_neg_run"] = longest_run(Wm < 0)
cand["wk_argmax"] = Wm.argmax(1).astype(float)
cand["wk_argmin"] = Wm.argmin(1).astype(float)
cand["wk_skew"] = pd.DataFrame(Wm).skew(axis=1).values
cand["wk_kurt"] = pd.DataFrame(Wm).kurt(axis=1).values
cand["wk_autocorr1"] = np.array([np.corrcoef(r[:-1], r[1:])[0,1] if r.std()>0 else 0 for r in Wm])
cum = np.cumsum(Wm, 1)
cand["cum_drawdown"] = (np.maximum.accumulate(cum,1) - cum).max(1)  # worst dip from peak
cand["cum_argmax"] = cum.argmax(1).astype(float)
cand["cum_final_over_peak"] = np.divide(cum[:,-1], np.abs(cum).max(1)+1e-9)
# (B) daily-activity temporal pattern
dd = np.diff(Dm, axis=1)
cand["day_sign_changes"] = (np.diff(np.sign(Dm-Dm.mean(1,keepdims=True)),axis=1)!=0).sum(1)
cand["day_longest_high_run"] = longest_run(Dm > np.median(Dm,1,keepdims=True))
cand["day_argmax"] = Dm.argmax(1).astype(float)
cand["day_trend_slope"] = np.array([np.polyfit(np.arange(16), r, 1)[0] for r in Dm])
cand["day_last4_minus_first4"] = Dm[:,-4:].mean(1) - Dm[:,:4].mean(1)
cand["day_entropy"] = -np.sum((p:=np.abs(Dm)/(np.abs(Dm).sum(1,keepdims=True)+1e-9))*np.log(p+1e-9),1)
# (C) cross-block interactions (activity x grade dynamics x completion)
cr = (tr["tugas_selesai"]/tr["tugas_diberikan"].replace(0,np.nan)).fillna(0).values
cand["daymean_x_wkstd"] = Dm.mean(1) * Wm.std(1)
cand["daystd_x_compl"] = Dm.std(1) * cr
cand["tryout_x_wkstd"] = tr["skor_tryout"].values * Wm.std(1)
cand["attend_x_compl"] = tr["indeks_kehadiran"].values * cr
cand["motiv_x_disc"] = tr["skor_motivasi"].values * tr["skor_kedisiplinan"].values
cand["literasi_x_minat"] = tr["skor_literasi"].values * tr["skor_minat_belajar"].values
# (D) distributional / rank position within cohort
for c in ["skor_tryout","indeks_kehadiran","skor_literasi","skor_motivasi"]:
    cand[f"{c}_pctile"] = rankdata(tr[c].values)/len(tr)

C = pd.DataFrame(cand).replace([np.inf,-np.inf],np.nan).fillna(0.0)

rows = []
for name in C.columns:
    v = C[name].values
    sp = spearmanr(v, yv).correlation
    # partial: correlation of feature-residual (vs cur_rank) with target-residual
    fr = rankdata(v) - np.polyval(np.polyfit(cur_rank, rankdata(v), 1), cur_rank)
    tr_ = rankdata(yv) - np.polyval(np.polyfit(cur_rank, rankdata(yv), 1), cur_rank)
    partial = spearmanr(fr, tr_).correlation
    rows.append((name, sp, partial))
res = pd.DataFrame(rows, columns=["feature","spearman","partial_vs_current"]).sort_values(
    "partial_vs_current", key=lambda s: s.abs(), ascending=False)
print("\n=== candidate signal screen (partial = signal orthogonal to current) ===")
print(res.round(4).to_string(index=False))

# ---- linear probe: does a Ridge direction on FULL scaled features add orthogonal signal?
def linear_oof(seed=42):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed); s=np.zeros(len(y))
    for tr_i,va_i in cv.split(Xcur,y):
        sc=StandardScaler().fit(Xcur.iloc[tr_i]); Xt=sc.transform(Xcur.iloc[tr_i]); Xv=sc.transform(Xcur.iloc[va_i])
        m=Ridge(alpha=20.0).fit(Xt,y.iloc[tr_i]); s[va_i]=m.predict(Xv)
    return s
lin=rankdata(linear_oof())
fr=lin-np.polyval(np.polyfit(cur_rank,lin,1),cur_rank)
tr_=rankdata(yv)-np.polyval(np.polyfit(cur_rank,rankdata(yv),1),cur_rank)
print(f"\nlinear-probe (Ridge) rank: spearman={spearmanr(lin,yv).correlation:.4f}  "
      f"partial_vs_current={spearmanr(fr,tr_).correlation:.4f}")

top = res.head(10)["feature"].tolist()
Path("outputs/exp_signal_screen.json").write_text(json.dumps({
    "seeds_note":"screen on seed42 ordinal rank","current_nfeat":int(Xcur.shape[1]),
    "screen":res.round(5).to_dict(orient="records"),
    "linear_probe_partial":float(spearmanr(fr,tr_).correlation),
    "top_orthogonal":top,"no_kaggle_submission_made":True},indent=2))
print("\ntop orthogonal candidates:",top)
print("wrote outputs/exp_signal_screen.json")
