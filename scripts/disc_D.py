"""Discovery batch D (refine): denoised daily lag-3 persistence + weekly persistence.

First pass found:
  D_pacf3 (daily Durbin-Levinson PACF lag3)  partial +0.053
  WD_wk_persist_plus_dly_lag3                partial +0.047
The daily period-3 oscillation is nominally 'captured' by v6, but the PACF-lag3
representation appears to carry residual orthogonal signal (a denoised estimate).

Refine: multiple estimators of the SAME daily lag-3 quantity + weekly lag-1
persistence, combined and shrunk, to see how high the orthogonal partial can go
and confirm it is a robust NEW representation vs noise.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/github/datathon2026_playground"
TR = f"{ROOT}/kaggle/train.csv"
OOF = f"{ROOT}/outputs/v6_oof.npz"
OUT = f"{ROOT}/outputs/disc_D.json"

WEEK_COLS = [f"nilai_minggu_{i:02d}" for i in range(1, 13)]
DAY_COLS = [f"aktivitas_hari_{i:02d}" for i in range(1, 17)]


def center(M):
    return M - M.mean(axis=1, keepdims=True)


def _fix(v):
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v


def autocov(x, k):
    T = len(x)
    return (x[: T - k] * x[k:]).sum() / T


def raw_ac(M, k):
    """Raw (biased) autocorr at lag k, per row."""
    n = M.shape[0]
    out = np.zeros(n)
    for i in range(n):
        x = M[i]
        c0 = autocov(x, 0)
        out[i] = autocov(x, k) / (c0 + 1e-12)
    return _fix(out)


def durbin_levinson_lag(M, maxlag):
    """PACF at each lag via Durbin-Levinson; return dict lag->vec."""
    n, T = M.shape
    pac = {k: np.zeros(n) for k in range(1, maxlag + 1)}
    for i in range(n):
        x = M[i]
        r = np.array([autocov(x, k) for k in range(0, maxlag + 1)])
        if r[0] < 1e-12:
            continue
        rho = r / r[0]
        phi_prev = np.zeros(maxlag + 1)
        v = 1.0
        for k in range(1, maxlag + 1):
            acc = rho[k]
            for j in range(1, k):
                acc -= phi_prev[j] * rho[k - j]
            refl = acc / (v + 1e-12)
            phi = phi_prev.copy()
            phi[k] = refl
            for j in range(1, k):
                phi[j] = phi_prev[j] - refl * phi_prev[k - j]
            v = v * (1 - refl * refl)
            pac[k][i] = refl
            phi_prev = phi
    for k in pac:
        pac[k] = _fix(pac[k])
    return pac


def ols_lag3(M):
    """Direct OLS: regress x_t on x_{t-3} only (period-3 persistence), per row."""
    x0 = M[:, :-3]
    x3 = M[:, 3:]
    num = (x0 * x3).sum(axis=1)
    den = (x0 * x0).sum(axis=1)
    return _fix(num / (den + 1e-12))


def full_ar3_coef3(M):
    """Full OLS AR(3): x_t ~ a1 x_{t-1}+a2 x_{t-2}+a3 x_{t-3}; return a3 (the
    part of lag-3 dependence NOT explained by lags 1-2 = cleanest period-3)."""
    n, T = M.shape
    a3 = np.zeros(n)
    for i in range(n):
        x = M[i]
        y = x[3:]
        X = np.column_stack([x[2:-1], x[1:-2], x[:-3]])
        A = X.T @ X
        b = X.T @ y
        try:
            sol = np.linalg.solve(A + 1e-9 * np.eye(3), b)
            a3[i] = sol[2]
        except np.linalg.LinAlgError:
            pass
    return _fix(a3)


def z(v):
    v = _fix(v)
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    d = np.load(OOF)
    cur = d["oof_rank"].astype(float)
    y = d["y"].astype(float)
    ry = rankdata(y)
    ty = ry - np.polyval(np.polyfit(cur, ry, 1), cur)
    tvar = np.sqrt((ty ** 2).mean())

    def partial(v):
        v = _fix(v)
        rv = rankdata(v)
        fr = rv - np.polyval(np.polyfit(cur, rv, 1), cur)
        fv = np.sqrt((fr ** 2).mean())
        return 0.0 if fv <= 1e-9 else float((fr * ty).mean() / (fv * tvar + 1e-12))

    ryy = rankdata(y)
    b = ryy - ryy.mean()

    def spear(v):
        v = _fix(v)
        a = rankdata(v)
        a = a - a.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

    tr = pd.read_csv(TR)
    W = center(tr[WEEK_COLS].to_numpy(float))
    Dm = center(tr[DAY_COLS].to_numpy(float))

    dpac = durbin_levinson_lag(Dm, 3)
    wpac = durbin_levinson_lag(W, 3)

    cands = {}
    # daily lag-3 estimators (different denoisings of period-3 persistence)
    cands["D_pacf3"] = dpac[3]
    cands["D_rawac3"] = raw_ac(Dm, 3)
    cands["D_olslag3"] = ols_lag3(Dm)
    cands["D_ar3_a3"] = full_ar3_coef3(Dm)
    # weekly lag-1 persistence (the captured weekly structure) for combining
    W_olsar1 = _fix((W[:, :-1] * W[:, 1:]).sum(1) / ((W[:, :-1] ** 2).sum(1) + 1e-12))
    cands["W_olsar1"] = W_olsar1
    cands["W_pacf1"] = wpac[1]

    # combined weekly-persist + daily-lag3 (multiple estimators of daily lag3)
    cands["WD_wkar1_plus_dpacf3"] = z(W_olsar1) + z(dpac[3])
    cands["WD_wkar1_plus_dols3"] = z(W_olsar1) + z(cands["D_olslag3"])
    cands["WD_wkar1_plus_dar3a3"] = z(W_olsar1) + z(cands["D_ar3_a3"])
    # average of the daily-lag3 estimators = lower-noise latent lag3 persistence
    d3_avg = z(dpac[3]) + z(cands["D_olslag3"]) + z(cands["D_ar3_a3"])
    cands["D_lag3_avg3est"] = d3_avg
    cands["WD_wkar1_plus_d3avg"] = z(W_olsar1) + z(d3_avg)
    # weekly pacf variants combined
    cands["WD_wkpacf1_plus_dpacf3"] = z(wpac[1]) + z(dpac[3])
    # try weighting daily-lag3 more (it had the higher solo partial)
    cands["WD_1w_2d3"] = z(W_olsar1) + 2.0 * z(dpac[3])
    cands["WD_2w_1d3"] = 2.0 * z(W_olsar1) + z(dpac[3])

    rows = [(n, spear(v), partial(v)) for n, v in cands.items()]
    rows.sort(key=lambda r: -abs(r[2]))
    print(f"{'name':30s} {'spearman':>10s} {'partial':>10s}")
    print("-" * 54)
    for n, s, p in rows:
        flag = "  <== INTERESTING" if abs(p) >= 0.04 else ""
        print(f"{n:30s} {s:10.4f} {p:10.4f}{flag}")

    with open(OUT, "w") as f:
        json.dump({"candidates": [{"name": n, "spearman": s, "partial": p} for n, s, p in rows],
                   "no_kaggle_submission_made": True}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
