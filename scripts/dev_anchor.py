"""DEV only: anchored planner (offset = FullProfile log-odds, shrunken corrections) replayed on stored scores."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as ap
from amifx.config import AGENTS, FEEDBACK
from amifx.data.loader import load
from amifx.evaluate import load_run
from amifx.planner import logit

def fit_offset(X, off, y, lam):
    n, k = X.shape
    def f(b):
        z = off + X @ b[:k] + b[k]; p = 1 / (1 + np.exp(-z))
        ll = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean()
        g = X.T @ (p - y) / n; gb = (p - y).mean()
        return ll + lam * (b[:k] ** 2).sum() / 2, np.r_[g + lam * b[:k], gb]
    return minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B").x

ds, tagv = sys.argv[1], sys.argv[2]
rows = []
for seed in [0, 1]:
    df, info = load_run(f"{ds}__dev__{tagv}__s{seed}"); D = load(ds); sd = info["split_def"]; fb = FEEDBACK["realistic"]
    stp = df.step.values; y = df.y.values.astype(int); yn = df.y_noisy.values.astype(int)
    rev = df["r:AMIF-X"].values; init = stp <= sd["initial_end"]; yo = np.where(init, y, yn)
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]; L = logit(P)
    res = {lam: np.full(len(df), np.nan) for lam in [0.001, 0.01, 0.1, 1.0]}
    for tt in range(sd["test_start"], sd["last"] + 1):
        h = (init | (rev & (stp <= tt - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == tt
        X = np.hstack([L[:, 1:], Z]); mu, s = X[h].mean(0), X[h].std(0) + 1e-6; Xs = (X - mu) / s
        for lam in res:
            b = fit_offset(Xs[h], L[h, 0], yo[h], lam); res[lam][cur] = L[cur, 0] + Xs[cur] @ b[:-1] + b[-1]
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    r = {"seed": seed, "fullprofile": 100 * ap(y[te], P[te, 0]), "stored": 100 * ap(y[te], df["g:AMIF-X"].values[te])}
    r.update({f"anchor_lam{l}": 100 * ap(y[te], v[te]) for l, v in res.items()}); rows.append(r)
T = pd.DataFrame(rows); print(T.round(2).to_string()); print(T.mean().round(2).to_string())
