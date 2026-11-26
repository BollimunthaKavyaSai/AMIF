"""DEV only: online (prequential) selection between learned and reliability-anchored planners."""
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
W = int(sys.argv[3]) if len(sys.argv) > 3 else 5
for seed in [0, 1]:
    df, info = load_run(f"{ds}__dev__{tagv}__s{seed}"); sd = info["split_def"]
    stp = df.step.values; y = df.y.values.astype(int); init = stp <= sd["initial_end"]
    yo = np.where(init, y, df.y_noisy.values.astype(int)); rev = df["r:AMIF-X"].values
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]
    S = {k: np.full(len(df), np.nan) for k in ["learned", "anchor", "selected"]}; pick = []
    for t in range(sd["initial_end"] + 1, sd["last"] + 1):
        h = (init | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == t
        S["learned"][cur] = Planner().fit(P[h], yo[h], Z[h]).score(P[cur], Z[cur])
        j = int(np.argmax([ap(yo[h], P[h, k]) for k in range(len(AGENTS))])); order = [j] + [k for k in range(len(AGENTS)) if k != j]
        S["anchor"][cur] = Planner(kind="anchor").fit(P[h][:, order], yo[h], Z[h]).score(P[cur][:, order], Z[cur])
        # prequential comparison on the most recent W stream steps with available (recorded) labels
        v = h & ~init & (stp >= t - 1 - fb.delay - W + 1) & ~np.isnan(S["learned"])
        if v.sum() > 20 and 0 < yo[v].sum() < v.sum():
            choice = "learned" if ap(yo[v], S["learned"][v]) >= ap(yo[v], S["anchor"][v]) else "anchor"
        else:
            choice = "learned"
        S["selected"][cur] = S[choice][cur]; pick.append(choice[0])
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    rows.append({"seed": seed, **{k: round(100 * ap(y[te], v[te]), 2) for k, v in S.items()}, "picks": "".join(pick)})
print(pd.DataFrame(rows).to_string())
