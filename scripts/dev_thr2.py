"""DEV only: threshold from the CURRENT planner applied to history (consistent calibration) vs stored rule."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from amifx.config import AGENTS, FEEDBACK, PRED_DIR
from amifx.data.loader import load
from amifx.evaluate import load_run, decisions, test_part
from amifx.planner import Planner, best_f1_threshold
from amifx import stats as st
ds = sys.argv[1]; fb = FEEDBACK["realistic"]; D = load(ds); rows = []
for f in sorted(PRED_DIR.glob(f"{ds}__dev__realistic*__s*.parquet")):
    if "devbase" in f.stem: continue
    df, info = load_run(f.stem); sd = info["split_def"]; stp = df.step.values; y = df.y.values.astype(int)
    yo = np.where(stp <= sd["initial_end"], y, df.y_noisy.values.astype(int)); rev = df["r:AMIF-X"].values
    P = np.column_stack([df[f"c:AMIF-X/{a}"].values for a in AGENTS]); Z = D.ctx()[df.row.values]
    kind = info.get("amx", {}).get("planner", "learned")
    d = np.zeros(len(df), int)
    for t in range(sd["test_start"], sd["last"] + 1):
        h = ((stp <= sd["initial_end"]) | (rev & (stp <= t - 1 - fb.delay))) & ~np.isnan(P).any(1); cur = stp == t
        pl = Planner(kind=kind).fit(P[h], yo[h], Z[h])
        d[cur] = pl.score(P[cur], Z[cur]) >= best_f1_threshold(yo[h], pl.score(P[h], Z[h]))
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    tp = test_part(df, info); ds_ = decisions(tp, info["thresholds"]["AMIF-X"], "g:AMIF-X")
    rows.append({"variant": f.stem.split("__")[2], "seed": f.stem[-1], "F1_stored_rule": 100 * st._prf(tp.y.values, ds_)[2],
                 "F1_current_planner_rule": 100 * st._prf(y[te], d[te])[2]})
print(pd.DataFrame(rows).groupby("variant").mean(numeric_only=True).round(2).to_string())
