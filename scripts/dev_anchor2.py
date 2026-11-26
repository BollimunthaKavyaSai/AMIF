"""DEV only: reliability-anchored planner (anchor = agent with best history PR-AUC) replayed on stored scores."""
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
    out = {k: np.full(len(df), np.nan) for k in ["learned", "anchor_best_hist", "anchor_best_recent"]}; anchors = []
    for t in range(sd["test_start"], sd["last"] + 1):
        h = (init | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == t
        out["learned"][cur] = Planner().fit(P[h], yo[h], Z[h]).score(P[cur], Z[cur])
        for nm, hh in [("anchor_best_hist", h), ("anchor_best_recent", h & (stp >= t - 6))]:
            if yo[hh].sum() < 5: hh = h
            j = int(np.argmax([ap(yo[hh], P[hh, k]) for k in range(len(AGENTS))]))
            order = [j] + [k for k in range(len(AGENTS)) if k != j]
            out[nm][cur] = Planner(kind="anchor").fit(P[h][:, order], yo[h], Z[h]).score(P[cur][:, order], Z[cur])
            if nm == "anchor_best_hist": anchors.append(AGENTS[j])
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    r = {"seed": seed, **{k: 100 * ap(y[te], v[te]) for k, v in out.items()}, "anchors": ",".join(sorted(set(anchors)))}
    rows.append(r)
print(pd.DataFrame(rows).to_string())
