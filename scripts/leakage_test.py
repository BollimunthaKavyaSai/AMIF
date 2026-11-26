"""Point-in-time verification (reviewer point 4).

Rebuild every case table after physically deleting all raw data later than step T, and
check that each feature of every case at step <= T is identical to the feature computed
from the full data. Any use of future information (full-data normalisation, cross-period
propagation, lifetime wallet totals...) would make some value differ.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amifx.config import CACHE, RESULTS  # noqa: E402
from amifx.data import amlsim, elliptic  # noqa: E402

res = {}
for name, T, builder in [("elliptic", 35, lambda: elliptic.build(force=True, max_step=35)),
                         ("elliptic", 42, lambda: elliptic.build(force=True, max_step=42)),
                         ("amlsim10k", 12, lambda: amlsim.build("amlsim10k", force=True, max_step=12))]:
    builder()
    full = pd.read_parquet(CACHE / f"{name}_cases.parquet")
    cut = pd.read_parquet(CACHE / f"{name}_cases_upto{T}.parquet")
    full = full[full.step <= T].set_index("id").sort_index()
    cut = cut.set_index("id").sort_index()
    assert full.index.equals(cut.index), "case sets differ"
    cols = [c for c in full.columns if "__" in c]
    a = full[cols].to_numpy("float64"); b = cut[cols].to_numpy("float64")
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    diff_cols = [c for c, ok in zip(cols, same.all(0)) if not ok]
    maxdiff = float(np.nanmax(np.abs(a - b))) if len(cols) else 0.0
    key = f"{name}_T{T}"
    res[key] = {"cases_checked": int(len(full)), "features_checked": len(cols), "values_checked": int(a.size),
                "identical_values": int(same.sum()), "columns_with_differences": diff_cols, "max_abs_diff": maxdiff}
    print(key, res[key], flush=True)
(RESULTS / "leakage_test.json").write_text(json.dumps(res, indent=1))
