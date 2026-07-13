"""Discovery A: Latent-var nonlinear & pairwise structure beyond known motiv*disc."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import combinations
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(ROOT, "kaggle", "train.csv"))
d = np.load(os.path.join(ROOT, "outputs", "v6_oof.npz"))
cur = d['oof_rank']; y = d['y']

ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty**2).mean())

def partial(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr**2).mean())
    return float((fr*ty).mean()/(fv*tvar+1e-12)) if fv > 1e-9 else 0.0

def spear(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    rv = rankdata(v)
    return float(np.corrcoef(rv, ry)[0, 1])

LAT = ["skor_motivasi","skor_kedisiplinan","skor_literasi","skor_minat_belajar",
       "indeks_kehadiran","skor_ekstrakurikuler","jarak_rumah_km","jumlah_saudara"]
L = {c: train[c].values.astype(float) for c in LAT}
compl = (train["tugas_selesai"] / train["tugas_diberikan"].replace(0, np.nan)).fillna(0).values
tryout = train["skor_tryout"].values.astype(float)

cands = {}

# squares, abs
for c in LAT:
    cands[f"{c}^2"] = L[c]**2
    cands[f"|{c}|"] = np.abs(L[c])

# all pairwise products, min, max, sum, diff, both>0 indicator
for a, b in combinations(LAT, 2):
    va, vb = L[a], L[b]
    cands[f"{a}*{b}"] = va*vb
    cands[f"min({a},{b})"] = np.minimum(va, vb)
    cands[f"max({a},{b})"] = np.maximum(va, vb)
    cands[f"both>0({a},{b})"] = ((va > 0) & (vb > 0)).astype(float)
    cands[f"sameSign({a},{b})"] = (np.sign(va) == np.sign(vb)).astype(float)

# 3-way products
for a, b, c in combinations(LAT, 3):
    cands[f"{a}*{b}*{c}"] = L[a]*L[b]*L[c]

# latent x completion, latent x tryout
for c in LAT:
    cands[f"{c}*compl"] = L[c]*compl
    cands[f"{c}*tryout"] = L[c]*tryout

# ---- Refinement pass ----
mo, di = L["skor_motivasi"], L["skor_kedisiplinan"]
ik, ek = L["indeks_kehadiran"], L["skor_ekstrakurikuler"]
def z(x): return (x - x.mean()) / (x.std() + 1e-12)
# nonlinear forms of the KNOWN product (is v6 missing curvature?)
cands["(motiv*disc)^2"] = (mo*di)**2
cands["|motiv*disc|"] = np.abs(mo*di)
cands["motiv^2*disc"] = mo**2 * di
cands["motiv*disc^2"] = mo * di**2
cands["rankmotiv*rankdisc"] = rankdata(mo)*rankdata(di)
cands["zmotiv*zdisc*sign"] = np.sign(mo*di)*np.sqrt(np.abs(mo*di))
# variants of the top residual pair
cands["z(ik)*z(ek)"] = z(ik)*z(ek)
cands["rankik*rankek"] = rankdata(ik)*rankdata(ek)
cands["-(ik*ek)"] = -(ik*ek)
cands["ik*ek*motiv"] = ik*ek*mo
# residual-of-residual: known product times other latents
for c in LAT:
    if c not in ("skor_motivasi","skor_kedisiplinan"):
        cands[f"(motiv*disc)*{c}"] = mo*di*L[c]

rows = []
for name, v in cands.items():
    rows.append((name, spear(v), partial(v)))

# also: which SINGLE pairwise product has the largest |raw spearman|?
prods = [(f"{a}*{b}", spear(L[a]*L[b])) for a,b in combinations(LAT,2)]
prods.sort(key=lambda r:-abs(r[1]))
print("\nTop-5 pairwise products by |raw spearman| (planted-product detector):")
for n,s in prods[:5]:
    print(f"  {n:45s} {s:8.4f}")
print()

rows.sort(key=lambda r: -abs(r[2]))
print(f"{'name':40s} {'spearman':>9s} {'partial':>9s}")
for name, sp, pa in rows[:40]:
    print(f"{name:40s} {sp:9.4f} {pa:9.4f}")

out = {"candidates": [{"name": n, "spearman": sp, "partial": pa} for n, sp, pa in rows],
       "no_kaggle_submission_made": True}
with open(os.path.join(ROOT, "outputs", "disc_A.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nTotal candidates:", len(rows))
