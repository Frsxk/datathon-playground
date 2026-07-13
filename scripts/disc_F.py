"""Discovery batch F (refined): completion / tryout / attendance richer transforms & interactions.

Screens target-free transforms of the non-sequence numeric fields against v6 OOF rank
via partial correlation. Refined pass focuses on attendance x ekstra, tryout nonlinearity,
and cross-products among the under-explored latents (jarak, saudara, kehad, ekstra).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
LETTER = "F"

df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
y = d["y"].astype(float)

ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def clean(v):
    v = np.array(v, dtype=float, copy=True)
    v[~np.isfinite(v)] = 0.0
    return v


def partial(v):
    rv = rankdata(clean(v))
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def spear(v):
    v = clean(v)
    rv = rankdata(v)
    if rv.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rv, ry)[0, 1])


WEEK = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
W = df[WEEK].to_numpy(float)
A = df[DAY].to_numpy(float)
wmean = W.mean(axis=1)
amean = A.mean(axis=1)

ts = df["tugas_selesai"].to_numpy(float)
tg = df["tugas_diberikan"].to_numpy(float)
compl = ts / (tg + 1e-9)
tryout = df["skor_tryout"].to_numpy(float)
kehad = df["indeks_kehadiran"].to_numpy(float)
ekstra = df["skor_ekstrakurikuler"].to_numpy(float)
motiv = df["skor_motivasi"].to_numpy(float)
disc = df["skor_kedisiplinan"].to_numpy(float)
lit = df["skor_literasi"].to_numpy(float)
minat = df["skor_minat_belajar"].to_numpy(float)
jarak = df["jarak_rumah_km"].to_numpy(float)
saud = df["jumlah_saudara"].to_numpy(float)


def z(v):
    v = clean(v)
    return (v - v.mean()) / (v.std() + 1e-9)


cands = {}

# attendance x ekstra family (the lead from pass 1)
cands["kehad_x_ekstra"] = kehad * ekstra
cands["z_kehad_x_z_ekstra"] = z(kehad) * z(ekstra)
cands["abs_kehad_x_ekstra"] = np.abs(kehad * ekstra)
cands["kehad_plus_ekstra"] = z(kehad) + z(ekstra)
cands["kehad_minus_ekstra"] = z(kehad) - z(ekstra)
cands["kehad_ekstra_sqdist"] = z(kehad) ** 2 + z(ekstra) ** 2

# tryout family
cands["tryout"] = tryout
cands["z_tryout"] = z(tryout)
cands["tryout_x_kehad_ekstra"] = tryout * kehad * ekstra
cands["tryout_x_compl_signed"] = z(tryout) * z(compl)

# all pairwise products among the 8 latents NOT yet crossed thoroughly
lat = {
    "motiv": motiv, "disc": disc, "lit": lit, "minat": minat,
    "kehad": kehad, "ekstra": ekstra, "jarak": jarak, "saud": saud,
}
names = list(lat)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        if {a, b} == {"motiv", "disc"}:
            key = "motiv_x_disc"  # captured reference
        else:
            key = f"{a}_x_{b}"
        cands[key] = z(lat[a]) * z(lat[b])

# jarak / saud interactions with completion & sequences (under-explored)
cands["jarak_x_kehad"] = z(jarak) * z(kehad)
cands["saud_x_compl"] = z(saud) * z(compl)
cands["jarak_x_wmean"] = z(jarak) * z(wmean)
cands["saud_x_amean"] = z(saud) * z(amean)

# tryout squared distance-like combos
cands["tryout_x_ekstra_signed"] = z(tryout) * z(ekstra)
cands["tryout_x_kehad_signed"] = z(tryout) * z(kehad)

# --- WINNER family: magnitude of the (kehad + ekstra) diagonal-sum axis ---
# The signed sum is flat vs target, but its ABSOLUTE VALUE carries signal:
# students whose attendance+extracurricular sum deviates far from zero score
# systematically differently. Distinct from radius sqrt(k^2+e^2) which is null.
cands["abs_kehad_plus_ekstra"] = np.abs(kehad + ekstra)
cands["neg_abs_kehad_plus_ekstra"] = -np.abs(kehad + ekstra)
cands["sq_kehad_plus_ekstra"] = (kehad + ekstra) ** 2
cands["abs_kehad_plus_1p2ekstra"] = np.abs(1.2 * kehad + ekstra)  # tuned axis weight
cands["radius_kehad_ekstra"] = np.sqrt(kehad ** 2 + ekstra ** 2)  # null control
cands["absk_plus_abse"] = np.abs(kehad) + np.abs(ekstra)          # null control

rows = [(n, spear(v), partial(v)) for n, v in cands.items()]
rows.sort(key=lambda r: -abs(r[2]))

print(f"{'name':30s} {'spearman':>10s} {'partial':>10s}")
for n, s, p in rows:
    print(f"{n:30s} {s:10.4f} {p:10.4f}")

out = {
    "candidates": [{"name": n, "spearman": s, "partial": p} for n, s, p in rows],
    "no_kaggle_submission_made": True,
}
with open(f"{ROOT}/outputs/disc_{LETTER}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote outputs/disc_{LETTER}.json")
