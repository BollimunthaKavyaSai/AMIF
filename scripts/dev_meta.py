"""DEV only: non-linear planners replayed on stored out-of-sample agent scores.
  lr       : current linear planner
  gbm      : LightGBM planner on agent log-odds + context
  gbm_aug  : LightGBM planner on agent log-odds + context + all raw evidence views (agents as meta-features)
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score as ap
from amifx.config import AGENTS, FEEDBACK
from amifx.data.loader import load
from amifx.evaluate import load_run
from amifx.planner import Planner, logit

ds, tagv = sys.argv[1], sys.argv[2]; scen = sys.argv[3] if len(sys.argv) > 3 else "realistic"
fb = FEEDBACK[scen]; D = load(ds); rows = []
Xall = D.X("all")
for seed in [0, 1]:
    df, info = load_run(f"{ds}__dev__{tagv}__s{seed}"); sd = info["split_def"]
    stp = df.step.values; y = df.y.values.astype(int); init = stp <= sd["initial_end"]
    yo = np.where(init, y, df.y_noisy.values.astype(int)); rev = df["r:AMIF-X"].values
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]
    L = logit(np.nan_to_num(P, nan=0.5)); R = Xall[df.row.values]
    out = {k: np.full(len(df), np.nan) for k in ["lr", "gbm", "gbm_aug"]}
    for t in range(sd["test_start"], sd["last"] + 1):
        h = (init | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == t
        out["lr"][cur] = Planner().fit(P[h], yo[h], Z[h]).score(P[cur], Z[cur])
        for nm, X in [("gbm", np.hstack([L, Z])), ("gbm_aug", np.hstack([L, Z, R]))]:
            m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=6)
            m.fit(X[h], yo[h]); out[nm][cur] = m.predict_proba(X[cur])[:, 1]
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    rows.append({"seed": seed, "FullProfile": round(100 * ap(y[te], P[te, 0]), 2),
                 **{k: round(100 * ap(y[te], v[te]), 2) for k, v in out.items()}})
T = pd.DataFrame(rows); print(T.to_string()); print(T.mean().round(2).to_string())
