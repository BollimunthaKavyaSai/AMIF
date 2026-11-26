"""Explanation validation (E1-E4) and measured runtime; writes results/*.json and tables."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amifx.config import RESULTS, TABLES  # noqa: E402
from amifx.evaluate import available  # noqa: E402
from amifx import explain, runtime, outage  # noqa: E402

what = sys.argv[1:] or ["e1", "e2", "e3", "e4", "rt", "outage"]
for ds, steps in [("elliptic", [38, 42, 46]), ("amlsim10k", [13, 17])]:
    if not available(ds, "main", "oracle"):
        continue
    out = {}
    p = RESULTS / f"{ds}_explain.json"
    if p.exists():
        out = json.loads(p.read_text())
    if "e1" in what:
        out["E1"] = explain.e1_exactness(ds, 0, steps[1]); print(ds, "E1", out["E1"], flush=True)
    if "e2" in what:
        agg, T = explain.e2_faithfulness(ds)
        T.to_csv(TABLES / f"{ds}_explain_faithfulness_seeds.csv", index=False)
        agg.columns = ["_".join(c) for c in agg.columns]; agg.to_csv(TABLES / f"{ds}_explain_faithfulness.csv")
        out["E2"] = agg.reset_index().to_dict("records"); print(ds, "E2\n", agg, flush=True)
    if "e3" in what:
        res = []
        for s in available(ds, "main", "oracle")[:3]:
            for t in steps:
                r, SH, TS, cases, rows = explain.e3_view_shapley(ds, s, t)
                res.append(r); print(ds, "E3", r, flush=True)
        out["E3"] = res
        pd.DataFrame(res).to_csv(TABLES / f"{ds}_explain_view_shapley.csv", index=False)
    if "e4" in what:
        out["E4"] = explain.e4_stability(ds); print(ds, "E4", out["E4"], flush=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    if "outage" in what:
        parts = [outage.run(ds, sd_, t) for sd_ in available(ds, "main", "oracle")[:3] for t in steps]
        O = pd.concat(parts); O.to_csv(TABLES / f"{ds}_outage_raw.csv", index=False)
        cols = [c for c in O.columns if c not in ("seed", "step", "outage")]
        S = O.groupby("outage")[cols].mean().round(3); S.to_csv(TABLES / f"{ds}_outage.csv"); print(ds, "OUTAGE"); print(S, flush=True)
    if "rt" in what:
        R = runtime.measure(ds, 0, steps[1])
        (RESULTS / f"{ds}_runtime.json").write_text(json.dumps(R, indent=1, default=str)); print(ds, "RT", R, flush=True)
