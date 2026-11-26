"""DEV only, batch 7: v15 + FullProfile agent on all evidence views (v17)."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import run
jobs = [(f"{ds}__dev__realistic+v17__s{s}", ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s),
         "--baselines", "none", "--suffix", "+v17", "--amx", json.dumps({"full_view": "all"})])
        for ds in ["elliptic", "amlsim10k"] for s in [0, 1]]
with ThreadPoolExecutor(1) as ex:
    for tag, st in ex.map(lambda j: run(j, 10), jobs):
        print(time.strftime("%H:%M:%S"), tag, st, flush=True)
