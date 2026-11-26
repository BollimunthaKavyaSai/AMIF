"""DEV only, batch 6: all-view relational agent (v15) and an additional collective agent (v16), on top of v13."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import run
V = {"v15": {"nr_view": "all"}, "v16": {"collective_agent": True}}
jobs = [(f"{ds}__dev__realistic+{v}__s{s}", ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s),
         "--baselines", "none", "--suffix", f"+{v}", "--amx", json.dumps(o)])
        for ds in ["elliptic", "amlsim10k"] for s in [0, 1] for v, o in V.items()]
with ThreadPoolExecutor(2) as ex:
    for tag, st in ex.map(lambda j: run(j, 5), jobs):
        print(time.strftime("%H:%M:%S"), tag, st, flush=True)
