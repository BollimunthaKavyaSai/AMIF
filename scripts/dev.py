"""Method development on the DEV split only (Elliptic++ steps 21-35 / AMLSim windows 6-10),
which ends before every test period. Design choices are frozen from these results before
any final test run.   python scripts/dev.py [elliptic|amlsim10k]"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import run  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else "elliptic"
SEEDS = [0, 1]
VARIANTS = {
    "v1": {},                                                            # CVCL v1 (committee review, consensus)
    "v2": {"explore": 0.0},                                              # top-risk review only
    "v3": {"explore": 0.0, "loo": True},                                 # + leave-one-view-out co-training
    "v4": {"explore": 0.0, "loo": True, "pseudo_w": 0.5},                # + down-weighted pseudo-labels
    "v5": {"explore": 0.0, "loo": True, "full_view": "all"},             # + FullProfile on all views
    "v6": {"explore": 0.0, "loo": True, "full_view": "all", "decay": True, "planner_decay": True},  # + recency
}
if len(sys.argv) > 2 and sys.argv[2] == "2":
    VARIANTS = {
        "v7": {"clean_hist": True},                                        # v1 + cleaned planner/threshold history
        "v8": {"clean_hist": True, "loo": True},                           # + leave-one-view-out co-training
        "v9": {"clean_hist": True, "explore": 0.1},                        # half the exploration budget
    }
if len(sys.argv) > 2 and sys.argv[2] == "3":
    VARIANTS = {
        "v10": {"clean_hist": True, "pseudo": "pos", "planner": "anchor"},   # positive-only pseudo-labels + anchored planner
        "v11": {"clean_hist": True, "pseudo": "none", "planner": "anchor"},  # noise filter only + anchored planner
    }
if ds.startswith("amlsim") and len(sys.argv) <= 2:
    VARIANTS = {"v1": {}, "v2": {"explore": 0.0}, "v7": {"clean_hist": True}}
jobs = []
for s in SEEDS:
    if len(sys.argv) > 2:
        pass
    else:
      jobs.append((f"{ds}__dev__realistic+devbase__s{s}",
                 ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s), "--no-agents",
                  "--baselines", "LR,RF,XGBoost,LightGBM,CatBoost,LightGBM-AllViews,LightGBM-Decay,XGBoost-AllViews,LR-AllViews,SGC",
                  "--suffix", "+devbase"]))
    if len(sys.argv) <= 2:
      jobs.append((f"{ds}__dev__realistic-nocvcl+v0__s{s}",
                 ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s), "--baselines", "none",
                  "--no-cvcl", "--suffix", "+v0", "--amx", json.dumps({"explore": 0.0})]))
    for v, opt in VARIANTS.items():
        jobs.append((f"{ds}__dev__realistic+{v}__s{s}",
                     ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s), "--baselines", "none",
                      "--suffix", f"+{v}", "--amx", json.dumps(opt)]))
workers = int(os.environ.get("AMIFX_WORKERS", 2)); threads = int(os.environ.get("AMIFX_THREADS", 5))
print(len(jobs), "dev jobs", flush=True)
with ThreadPoolExecutor(workers) as ex:
    for tag, st in ex.map(lambda j: run(j, threads), jobs):
        print(time.strftime("%H:%M:%S"), tag, st, flush=True)
