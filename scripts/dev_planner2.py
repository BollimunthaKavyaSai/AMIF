"""DEV only: planner history / regularisation variants replayed on stored scores."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from amifx.config import AGENTS, FEEDBACK
from amifx.data.loader import load
from amifx.evaluate import load_run
from amifx.planner import Planner
ds, tagv = sys.argv[1], sys.argv[2]; fb = FEEDBACK["realistic"]; D = load(ds); rows = []
for seed in [0, 1]:
    df, info = load_run(f"{ds}__dev__{tagv}__s{seed}"); sd = info["split_def"]
    stp = df.step.values; y = df.y.values.astype(int); init = stp <= sd["initial_end"]
    yo = np.where(init, y, df.y_noisy.values.astype(int)); rev = df["r:AMIF-X"].values
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]
    V = {"base_C1": (False, 1.0, True), "C0.1": (False, 0.1, True), "C0.01": (False, 0.01, True),
         "stream_only_C1": (True, 1.0, True), "stream_only_C0.1": (True, 0.1, True), "stream_only_noctx": (True, 1.0, False)}
    out = {k: np.full(len(df), np.nan) for k in V}
    for t in range(sd["test_start"], sd["last"] + 1):
        cur = stp == t
        for k, (so, C, ctx) in V.items():
            h = ((~init if so else init) | (rev & (stp <= t - 1 - fb.delay)) & ~init) & ~np.isnan(P).any(1)
            if so: h = rev & (stp <= t - 1 - fb.delay) & ~init & ~np.isnan(P).any(1)
            else: h = (init | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1)
            out[k][cur] = Planner(context=ctx, C=C).fit(P[h], yo[h], Z[h] if ctx else None).score(P[cur], Z[cur] if ctx else None)
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    rows.append({"seed": seed, "FullProfile": 100 * ap(y[te], P[te, 0]), **{k: 100 * ap(y[te], v[te]) for k, v in out.items()}})
T = pd.DataFrame(rows); print(T.round(2).to_string()); print(T.mean().round(2).to_string())
