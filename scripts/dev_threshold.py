"""DEV only: threshold rule comparison (all history vs unbiased history) for every model, post hoc."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from amifx.config import FEEDBACK, PRED_DIR
from amifx.data.loader import load
from amifx.evaluate import load_run
from amifx.planner import best_f1_threshold
from amifx import stats as st
ds = sys.argv[1]; fb = FEEDBACK["realistic"]; D = load(ds); rows = []
for f in sorted(PRED_DIR.glob(f"{ds}__dev__realistic*__s*.parquet")):
    df, info = load_run(f.stem); seed = int(f.stem.split("__s")[-1]); sd = info["split_def"]
    rng = np.random.default_rng(10_000 + seed); rng.random(D.n); ua = rng.random(D.n)[df.row.values]
    stp = df.step.values; y = df.y.values.astype(int); yn = df.y_noisy.values.astype(int); init = stp <= sd["initial_end"]
    for g in info["groups"]:
        if not (f.stem.split("__")[2].endswith("devbase") or g == "AMIF-X"):
            continue
        s = df["g:" + g].values; rev = df["r:" + g].values; aud = rev & (ua < fb.audit)
        yo = np.where(init, y, yn); res = {}
        for rule in ["all", "unbiased"]:
            d = np.zeros(len(df), int)
            for t in range(sd["test_start"], sd["last"] + 1):
                h = init | ((rev if rule == "all" else aud) & (stp <= t - 1 - fb.delay))
                d[stp == t] = s[stp == t] >= best_f1_threshold(yo[h], s[h])
            te = (stp >= sd["test_start"]) & (stp <= sd["last"])
            res[rule] = 100 * st._prf(y[te], d[te])[2]
        rows.append({"model": g if g != "AMIF-X" else f.stem.split("__")[2], "seed": seed, **res})
T = pd.DataFrame(rows).groupby("model")[["all", "unbiased"]].mean().round(2).sort_values("unbiased", ascending=False)
print(T.to_string())
