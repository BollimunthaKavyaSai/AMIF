"""DEV only, batch 4: collective baseline vs AMIF-X with cross-view relational propagation."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import run
jobs = []
for ds in ["amlsim10k", "elliptic"]:
    for s in [0, 1]:
        jobs.append((f"{ds}__dev__realistic+devcoll__s{s}", ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s),
                     "--no-agents", "--baselines", "LightGBM-Collective", "--suffix", "+devcoll"]))
        jobs.append((f"{ds}__dev__realistic+v12__s{s}", ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s),
                     "--baselines", "none", "--suffix", "+v12", "--amx", json.dumps({"nr_view": "all"})]))
with ThreadPoolExecutor(2) as ex:
    for tag, st in ex.map(lambda j: run(j, 5), jobs):
        print(time.strftime("%H:%M:%S"), tag, st, flush=True)
