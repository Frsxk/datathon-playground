#!/usr/bin/env python3
"""Cache the v6 out-of-fold latent rank so discovery scripts can screen NEW
signal by PARTIAL correlation (residual-vs-residual) against the current model
without refitting v6 each time. Averages OOF over seeds 42,43 to denoise.
Writes outputs/v6_oof.npz {oof_rank, y}. Cheap-ish (2 seeds x 5 folds x ord)."""
from __future__ import annotations
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from run_production_v6_temporal import feats, ord_score

def main():
    tr = pd.read_csv("kaggle/train.csv")
    y = tr["target"].to_numpy()
    X = feats(tr)
    seeds = (42, 43)
    oof_sum = np.zeros(len(y))
    for s in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr_i, va_i in cv.split(X, y):
            oof[va_i] = ord_score(X.iloc[tr_i], pd.Series(y).iloc[tr_i], X.iloc[va_i], s)
        oof_sum += pd.Series(oof).rank().to_numpy()
    oof_rank = oof_sum / len(seeds)
    Path("outputs").mkdir(exist_ok=True)
    np.savez("outputs/v6_oof.npz", oof_rank=oof_rank, y=y)
    from scipy.stats import spearmanr
    print("v6 OOF rank vs y spearman:", round(spearmanr(oof_rank, y).statistic, 4))
    print("wrote outputs/v6_oof.npz  n=", len(y))

if __name__ == "__main__":
    main()
