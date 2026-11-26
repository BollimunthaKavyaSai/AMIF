"""Summarise DEV-split results (method development only)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amifx.evaluate import load_run, decisions, test_part  # noqa: E402
from amifx.config import PRED_DIR, TABLES  # noqa: E402
from amifx import stats as st  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else "elliptic"
scen = sys.argv[2] if len(sys.argv) > 2 else "realistic"
rows = []
for f in sorted(PRED_DIR.glob(f"{ds}__dev__{scen}*__s*.parquet")):
    t = f.stem; df, info = load_run(t)
    te = test_part(df, info); var = t.split("__")[2]; seed = t.split("__s")[-1]
    for g in info["groups"]:
        if var.endswith(("devbase", "devcoll")) or g == "AMIF-X":
            d = decisions(te, info["thresholds"][g], "g:" + g)
            m = st.metrics(te.y.values, te["g:" + g].values, .5); p, r, f1, mcc, _ = st._prf(te.y.values, d)
            rows.append({"run": var, "model": g if var.endswith(("devbase", "devcoll")) else var, "seed": seed, "PR-AUC": m["PR-AUC"],
                         "ROC-AUC": m["ROC-AUC"], "F1": 100 * f1})
T = pd.DataFrame(rows).groupby("model")[["PR-AUC", "ROC-AUC", "F1"]].agg(["mean", "count"]).round(2)
T = T.sort_values(("PR-AUC", "mean"), ascending=False)
print(T.to_string())
T.to_csv(TABLES / f"dev_{ds}_{scen}.csv")
