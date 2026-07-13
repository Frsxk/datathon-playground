"""Discovery batch B - REFINE round: cross-sequence coupling gated by latents.

The main new structure found: coupling / lead-lag cross-correlation between the
daily-activity sequence and the weekly-grade sequence, gated by a latent var.
Top hits from disc_B.py:
  skor_minat_belajar * wd_leadminuslag   partial +0.046
  skor_literasi      * wd_coupling       partial +0.041

Both are genuinely orthogonal to v6 (bare autocorr) and to prior latent x autocorr
products (which never touched inter-sequence coupling). This round:
  - tries multiple resampling / alignment schemes for the two sequences,
  - sweeps lead/lag offsets,
  - screens ALL 8 latents against each coupling form,
  - runs a 4-fold stability probe (not just split-half) to gauge reliability,
  - looks for a combined 2-3 term representation.
Writes outputs/disc_B_refine.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]
LATENTS = [
    "skor_motivasi", "skor_kedisiplinan", "skor_literasi", "skor_minat_belajar",
    "indeks_kehadiran", "skor_ekstrakurikuler", "jarak_rumah_km", "jumlah_saudara",
]

df = pd.read_csv(f"{ROOT}/kaggle/train.csv")
y = df["target"].to_numpy().astype(float)
d = np.load(f"{ROOT}/outputs/v6_oof.npz")
cur = d["oof_rank"].astype(float)
ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def partial(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(v) < 1e-12:
        return 0.0
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def spearman(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(v), ry)[0, 1])


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


W = df[WEEK_COLS].to_numpy(float)
D = df[DAY_COLS].to_numpy(float)
Wc = W - W.mean(axis=1, keepdims=True)
Dc = D - D.mean(axis=1, keepdims=True)
lz = {k: z(df[k].to_numpy(float)) for k in LATENTS}


def resample_to(M, n):
    old = np.linspace(0, 1, M.shape[1])
    new = np.linspace(0, 1, n)
    return np.array([np.interp(new, old, row) for row in M])


def normed_cross(A, B, off):
    """Sum A[t]*B[t-off] normalized by norms. off>0 => B leads A."""
    if off > 0:
        a = A[:, off:]; b = B[:, :-off]
    elif off < 0:
        a = A[:, :off]; b = B[:, -off:]
    else:
        a = A; b = B
    num = (a * b).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1)) + 1e-9
    return num / den


results = []

# build coupling features under a few grids
grids = {"g12": 12, "g16": 16, "g8": 8}
coup_feats = {}
for gname, n in grids.items():
    Wr = resample_to(Wc, n)
    Dr = resample_to(Dc, n)
    for off in [-2, -1, 0, 1, 2]:
        coup_feats[f"{gname}_cc{off:+d}"] = normed_cross(Wr, Dr, off)
    # lead-minus-lag contrasts
    coup_feats[f"{gname}_lml1"] = normed_cross(Wr, Dr, 1) - normed_cross(Wr, Dr, -1)
    coup_feats[f"{gname}_lml2"] = normed_cross(Wr, Dr, 2) - normed_cross(Wr, Dr, -2)

# bare coupling features + latent-gated versions
cz = {k: z(v) for k, v in coup_feats.items()}
for k, v in cz.items():
    results.append((f"bare_{k}", spearman(v), partial(v)))
    for lk, lv in lz.items():
        results.append((f"{lk}__X__{k}", spearman(lv * v), partial(lv * v)))

results.sort(key=lambda r: -abs(r[2]))

# ---- 4-fold stability probe on top hits ----
name2vec = {}
for k, v in cz.items():
    name2vec[f"bare_{k}"] = v
    for lk, lv in lz.items():
        name2vec[f"{lk}__X__{k}"] = lv * v


def probe(name, v, K=4):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(v))
    folds = np.array_split(idx, K)
    ps = []
    for sub in folds:
        cs = cur[sub]
        rys = rankdata(y[sub]); tys = rys - np.polyval(np.polyfit(cs, rys, 1), cs)
        tvs = np.sqrt((tys ** 2).mean())
        rv = rankdata(v[sub]); fr = rv - np.polyval(np.polyfit(cs, rv, 1), cs)
        fv = np.sqrt((fr ** 2).mean())
        ps.append(float((fr * tys).mean() / (fv * tvs + 1e-12)) if fv > 1e-9 else 0.0)
    ps = np.array(ps)
    same = np.all(np.sign(ps) == np.sign(ps.mean()))
    print(f"PROBE {name:34s} full={partial(v):+.4f} folds={np.array2string(ps, precision=3, floatmode='fixed')} same_sign={same}")


print()
for name, sp, pa in results[:10]:
    probe(name, name2vec[name])

# ---- combined representation: sum of z-scored top orthogonal coupling terms ----
top_terms = [n for n, sp, pa in results if abs(pa) >= 0.035 and n.startswith(("skor", "indeks", "jarak", "jumlah"))][:6]
if len(top_terms) >= 2:
    combo = np.zeros(len(df))
    for n in top_terms:
        combo = combo + np.sign(partial(name2vec[n])) * z(name2vec[n])
    results.append(("COMBO_coupling_topterms", spearman(combo), partial(combo)))
    print()
    probe("COMBO_coupling_topterms", combo)

results.sort(key=lambda r: -abs(r[2]))
print(f"\n{'name':40s} {'spearman':>10s} {'partial':>10s}")
for name, sp, pa in results[:30]:
    print(f"{name:40s} {sp:10.4f} {pa:10.4f}")

out = {
    "candidates": [{"name": n, "spearman": s, "partial": p} for n, s, p in results],
    "no_kaggle_submission_made": True,
}
with open(f"{ROOT}/outputs/disc_B_refine.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote outputs/disc_B_refine.json")
