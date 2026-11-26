"""DEV only: post-hoc replay of planner variants on stored agent scores (no agent retraining)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import average_precision_score as ap  # noqa: E402

from amifx.config import AGENTS, FEEDBACK  # noqa: E402
from amifx.data.loader import load  # noqa: E402
from amifx.evaluate import load_run  # noqa: E402
from amifx.planner import Planner, logit  # noqa: E402


def nonneg_lr(X, y, w=None, C=1.0):
    n, k = X.shape; w = np.ones(n) if w is None else w
    def f(b):
        z = X @ b[:k] + b[k]; p = 1 / (1 + np.exp(-z))
        ll = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() / w.sum()
        g = X.T @ (w * (p - y)) / w.sum(); gb = (w * (p - y)).sum() / w.sum()
        return ll + (b[:k] ** 2).sum() / (2 * C * n), np.r_[g + b[:k] / (C * n), gb]
    r = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", bounds=[(0, None)] * k + [(None, None)])
    return r.x


ds, tagv = sys.argv[1], sys.argv[2]
rows = []
for seed in [0, 1]:
    t = f"{ds}__dev__{tagv}__s{seed}"
    df, info = load_run(t); D = load(ds)
    sd = info["split_def"]; fb = FEEDBACK["realistic"]
    rng = np.random.default_rng(10_000 + seed); rng.random(D.n); u_audit = rng.random(D.n)
    stp = df.step.values; y = df.y.values.astype(int); yn = df.y_noisy.values.astype(int)
    rev = df["r:AMIF-X"].values; aud = rev & (u_audit[df.row.values] < fb.audit)
    init = stp <= sd["initial_end"]
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]
    out = {k: np.full(len(df), np.nan) for k in ["lr_all", "lr_init", "lr_init+audit", "nonneg_all", "nonneg_init+audit",
                                                 "nonneg_init+audit_noctx", "fullprofile", "equal_top3"]}
    for tt in range(sd["test_start"], sd["last"] + 1):
        avail = init | (rev & (stp <= tt - 1 - fb.delay))
        cur = stp == tt
        yo = np.where(init, y, yn)
        H = {"all": avail, "init": init, "init+audit": init | (aud & (stp <= tt - 1 - fb.delay))}
        for nm in ["all", "init", "init+audit"]:
            h = H[nm] & ~np.isnan(P).any(1)
            pl = Planner().fit(P[h], yo[h], Z[h]); out["lr_" + nm if nm != "init+audit" else "lr_init+audit"][cur] = pl.score(P[cur], Z[cur])
        for nm, ctx in [("all", True), ("init+audit", True), ("init+audit", False)]:
            h = H[nm] & ~np.isnan(P).any(1)
            L = logit(P); mu, sdv = L[h].mean(0), L[h].std(0) + 1e-6
            X = (L - mu) / sdv
            if ctx:
                zm, zs = Z[h].mean(0), Z[h].std(0) + 1e-6; Xz = (Z - zm) / zs
                b = nonneg_lr(np.hstack([X[h], Xz[h]]), yo[h])          # context coefficients also >= 0 here
                s = np.hstack([X[cur], Xz[cur]]) @ b[:-1] + b[-1]
            else:
                b = nonneg_lr(X[h], yo[h]); s = X[cur] @ b[:-1] + b[-1]
            key = f"nonneg_{nm}" + ("" if ctx else "_noctx")
            out[key][cur] = s
        out["fullprofile"][cur] = P[cur, 0]
        out["equal_top3"][cur] = logit(P[cur][:, [0, 1, 7]]).mean(1)
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    r = {"seed": seed, "stored_fused": 100 * ap(y[te], df["g:AMIF-X"].values[te])}
    r.update({k: 100 * ap(y[te], v[te]) for k, v in out.items()})
    rows.append(r)
T = pd.DataFrame(rows); print(T.round(2).to_string()); print(T.mean().round(2).to_string())
