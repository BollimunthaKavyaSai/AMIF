"""DEV only: planner variants replayed on stored OOS agent scores (light memory: reads only ctx columns)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import average_precision_score as ap
from amifx.config import AGENTS, CACHE, FEEDBACK
from amifx.evaluate import load_run
from amifx.planner import Planner, logit

ds, tagv = sys.argv[1], sys.argv[2]; fb = FEEDBACK["realistic"]
cols = [c for c in pq.read_schema(CACHE / f"{ds}_cases.parquet").names if c.startswith("ctx__")]
CTX = pd.read_parquet(CACHE / f"{ds}_cases.parquet", columns=cols).to_numpy("float32")
rows = []
for seed in [0, 1]:
    df, info = load_run(f"{ds}__dev__{tagv}__s{seed}"); sd = info["split_def"]
    stp = df.step.values; y = df.y.values.astype(int); init = stp <= sd["initial_end"]
    yo = np.where(init, y, df.y_noisy.values.astype(int)); rev = df["r:AMIF-X"].values
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = CTX[df.row.values]
    out = {k: np.full(len(df), np.nan) for k in ["learned", "anchor", "select", "noctx", "top3"]}
    for t in range(sd["initial_end"] + 1, sd["last"] + 1):
        h = (init | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == t
        out["learned"][cur] = Planner().fit(P[h], yo[h], Z[h]).score(P[cur], Z[cur])
        out["noctx"][cur] = Planner(context=False).fit(P[h], yo[h]).score(P[cur])
        apk = [ap(yo[h], P[h, k]) for k in range(len(AGENTS))]
        j = int(np.argmax(apk)); order = [j] + [k for k in range(len(AGENTS)) if k != j]
        out["anchor"][cur] = Planner(kind="anchor").fit(P[h][:, order], yo[h], Z[h]).score(P[cur][:, order], Z[cur])
        top = list(np.argsort(apk)[::-1][:3])
        out["top3"][cur] = Planner().fit(P[h][:, top], yo[h], Z[h]).score(P[cur][:, top], Z[cur])
        v = h & ~init & (stp >= t - 1 - fb.delay - 4) & ~np.isnan(out["learned"])
        ch = "learned"
        if v.sum() > 20 and 0 < yo[v].sum() < v.sum() and ap(yo[v], out["anchor"][v]) > ap(yo[v], out["learned"][v]):
            ch = "anchor"
        out["select"][cur] = out[ch][cur]
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    rows.append({"seed": seed, "FullProfile": 100 * ap(y[te], P[te, 0]), **{k: 100 * ap(y[te], v[te]) for k, v in out.items()}})
T = pd.DataFrame(rows); print(ds, tagv); print(T.round(2).to_string()); print(T.mean().round(2).to_string())
