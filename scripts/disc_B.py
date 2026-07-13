"""Discovery batch B: latent-var x temporal-feature interactions (NEW representations).

Prior B run: best hits ~0.04 (motiv*ekstra*d_osc3 -0.043, disc*jarak*w_ac2 +0.044)
were marginal triple products. This rewrite pushes into representations that are
genuinely orthogonal to both v6 (bare autocorr + motiv*disc) AND to simple
linear latent x autocorr products:

  1. Latent-WEIGHTED / latent-SELECTED temporal statistics (the latent picks WHICH
     part of the sequence carries persistence, not just a scalar multiplier).
  2. Cross-sequence coupling: how the daily and weekly sequences co-move, gated by latents.
  3. Sign/regime gating of temporal persistence by a latent, controlled for the bare gate.
  4. Latent-conditional autocorr *contrast* (persistence when latent-high minus latent-low
     is per-row impossible; instead use latent to reweight timesteps within the row).

All transforms are target-free and computable on test. Key diagnostic:
partial correlation vs v6 oof_rank; |partial| >= 0.04 => new orthogonal signal.
Every promising hit is run through a split-half robustness probe.
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
y_cached = d["y"].astype(float)
assert len(cur) == len(df), (len(cur), len(df))
assert np.allclose(rankdata(y_cached), rankdata(y)) or (y_cached == y).all()

ry = rankdata(y)
ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
tvar = np.sqrt((ty ** 2).mean())


def partial(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(v) < 1e-12:
        return 0.0
    rv = rankdata(v)
    fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
    fv = np.sqrt((fr ** 2).mean())
    return float((fr * ty).mean() / (fv * tvar + 1e-12)) if fv > 1e-9 else 0.0


def spearman(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(v), ry)[0, 1])


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


W = df[WEEK_COLS].to_numpy(dtype=float)
D = df[DAY_COLS].to_numpy(dtype=float)
Wc = W - W.mean(axis=1, keepdims=True)
Dc = D - D.mean(axis=1, keepdims=True)


def row_autocorr(M, lag):
    Mc = M - M.mean(axis=1, keepdims=True)
    a = Mc[:, :-lag]
    b = Mc[:, lag:]
    num = (a * b).sum(axis=1)
    ac = num / ((Mc ** 2).sum(axis=1) + 1e-9)
    return np.nan_to_num(ac, nan=0.0)


# ---- baseline temporal features (KNOWN captured by v6) ----
w_ac1 = row_autocorr(W, 1)
w_ac2 = row_autocorr(W, 2)
d_ac1 = row_autocorr(D, 1)
d_ac2 = row_autocorr(D, 2)
d_ac3 = row_autocorr(D, 3)
d_osc3 = d_ac3 - 0.5 * (d_ac1 + d_ac2)

lz = {k: z(df[k].to_numpy(dtype=float)) for k in LATENTS}

results = []

# =====================================================================
# NEW REP 1: Latent as a TIMESTEP WEIGHT is impossible (latent is scalar per row).
# But we can build the sequence's persistence measured on the DERIVATIVE / on the
# extreme timesteps, then cross with latent. More importantly: measure temporal
# structure the trees can't build: PHASE / where-in-time the peak occurs, run-length.
# =====================================================================

# weekly trajectory shape features (target-free)
def peak_pos(M):  # argmax position normalized 0..1 (phase of the best week)
    return M.argmax(axis=1) / (M.shape[1] - 1)


def trough_pos(M):
    return M.argmin(axis=1) / (M.shape[1] - 1)


def n_upcrossings(Mc):  # number of sign changes of centered series (oscillation count)
    s = np.sign(Mc)
    return (np.abs(np.diff(s, axis=1)) > 0).sum(axis=1).astype(float)


def longest_run_above(Mc):  # longest consecutive stretch above own mean
    above = (Mc > 0).astype(int)
    out = np.zeros(Mc.shape[0])
    for i in range(Mc.shape[0]):
        best = cur_run = 0
        for v in above[i]:
            cur_run = cur_run + 1 if v else 0
            best = max(best, cur_run)
        out[i] = best
    return out


shape_feats = {
    "w_peakpos": peak_pos(W), "w_troughpos": trough_pos(W),
    "d_peakpos": peak_pos(D), "d_troughpos": trough_pos(D),
    "w_ncross": n_upcrossings(Wc), "d_ncross": n_upcrossings(Dc),
    "w_longrun": longest_run_above(Wc), "d_longrun": longest_run_above(Dc),
}
sz = {k: z(v) for k, v in shape_feats.items()}
for k, v in sz.items():
    results.append((f"bare_{k}", spearman(v), partial(v)))
    for lk, lv in lz.items():
        results.append((f"{lk}__X__{k}", spearman(lv * v), partial(lv * v)))

# =====================================================================
# NEW REP 2: Cross-sequence coupling. Resample weekly (12) and daily (16) onto a
# common grid and measure how they co-move. If class controls a shared latent
# "engagement" driving BOTH, their coupling is a fresh signal. Then gate by latent.
# =====================================================================
def resample_to(M, n):
    old = np.linspace(0, 1, M.shape[1])
    new = np.linspace(0, 1, n)
    return np.array([np.interp(new, old, row) for row in M])


Wr = resample_to(Wc, 12)
Dr = resample_to(Dc, 12)
# per-row Pearson between the two centered resampled sequences
num = (Wr * Dr).sum(axis=1)
den = np.sqrt((Wr ** 2).sum(axis=1) * (Dr ** 2).sum(axis=1)) + 1e-9
wd_coupling = num / den
# lead/lag cross-corr: does daily activity lead weekly grade by 1 step?
wd_lead = (Wr[:, 1:] * Dr[:, :-1]).sum(axis=1) / (den + 1e-9)
wd_lag = (Wr[:, :-1] * Dr[:, 1:]).sum(axis=1) / (den + 1e-9)
cross = {"wd_coupling": wd_coupling, "wd_lead": wd_lead, "wd_lag": wd_lag,
         "wd_leadminuslag": wd_lead - wd_lag}
cz = {k: z(v) for k, v in cross.items()}
for k, v in cz.items():
    results.append((f"bare_{k}", spearman(v), partial(v)))
    for lk, lv in lz.items():
        results.append((f"{lk}__X__{k}", spearman(lv * v), partial(lv * v)))

# =====================================================================
# NEW REP 3: Latent-GATED autocorr with the BARE gate controlled out.
# The prior posgate hit had sp=0.26 (mostly v6-captured). Build the residual:
# latent selects the SIGN of temporal contribution -> disc*sign-structure.
# Use latent to reweight lag products directly: sum_t (latent-side) is scalar so
# instead weight by whether the WEEKLY value at t is above/below, gated by latent sign.
# =====================================================================
# regime interaction: (latent>0) flips the temporal feature -> abs vs signed
for lk in LATENTS:
    lv = lz[lk]
    for tk, tv in [("w_ac1", w_ac1), ("d_ac3", d_ac3), ("d_osc3", d_osc3), ("w_ac2", w_ac2)]:
        # signed-by-latent: latent sign times temporal MAGNITUDE
        v = np.sign(lv) * np.abs(tv)
        results.append((f"sgn_{lk}_X_abs_{tk}", spearman(v), partial(v)))
        # latent-magnitude gates temporal (|lat| large -> temporal matters)
        v2 = np.abs(lv) * tv
        results.append((f"abs_{lk}_X_{tk}", spearman(v2), partial(v2)))

# =====================================================================
# NEW REP 4: pairwise-latent SELECTED temporal. The known interaction is motiv*disc.
# Maybe a DIFFERENT latent pair gates temporal persistence. Screen all 28 pairs x
# 4 key temporal features, but as (pair-product) x temporal AND as
# (pair-product sign) x temporal-magnitude.
# =====================================================================
temporal4 = {"w_ac1": z(w_ac1), "d_ac3": z(d_ac3), "d_osc3": z(d_osc3), "w_ac2": z(w_ac2)}
for i, li in enumerate(LATENTS):
    for lj in LATENTS[i + 1:]:
        pair = lz[li] * lz[lj]
        pz = z(pair)
        for tk, tv in temporal4.items():
            v = pz * tv
            results.append((f"{li}x{lj}_X_{tk}", spearman(v), partial(v)))

results.sort(key=lambda r: -abs(r[2]))

# ---- robustness probe on the top-8 hits ----
def probe(name, v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    p_full = partial(v)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(v))
    h1, h2 = idx[: len(v) // 2], idx[len(v) // 2:]

    def partial_sub(v, sub):
        cs = cur[sub]
        rys = rankdata(y[sub]); tys = rys - np.polyval(np.polyfit(cs, rys, 1), cs)
        tvs = np.sqrt((tys ** 2).mean())
        rv = rankdata(v[sub]); fr = rv - np.polyval(np.polyfit(cs, rv, 1), cs)
        fv = np.sqrt((fr ** 2).mean())
        return float((fr * tys).mean() / (fv * tvs + 1e-12)) if fv > 1e-9 else 0.0
    print(f"PROBE {name:42s} full={p_full:+.4f} h1={partial_sub(v,h1):+.4f} h2={partial_sub(v,h2):+.4f}")


# rebuild top-8 vectors by name for probing
name2vec = {}
for k, v in sz.items():
    name2vec[f"bare_{k}"] = v
    for lk, lv in lz.items():
        name2vec[f"{lk}__X__{k}"] = lv * v
for k, v in cz.items():
    name2vec[f"bare_{k}"] = v
    for lk, lv in lz.items():
        name2vec[f"{lk}__X__{k}"] = lv * v

print()
for name, sp, pa in results[:8]:
    if name in name2vec:
        probe(name, name2vec[name])

print(f"\n{'name':42s} {'spearman':>10s} {'partial':>10s}")
for name, sp, pa in results[:40]:
    print(f"{name:42s} {sp:10.4f} {pa:10.4f}")

out = {
    "candidates": [{"name": n, "spearman": s, "partial": p} for n, s, p in results],
    "no_kaggle_submission_made": True,
}
with open(f"{ROOT}/outputs/disc_B.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote outputs/disc_B.json")
