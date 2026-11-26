"""DEV only, batch 5: Graph agent with a stable tree learner on propagated features (v13), with/without CVCL (v14)."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import run
V = {"v13": ([], {"graph_model": "lgbm"}), "v14": (["--cvcl"], {"graph_model": "lgbm", "clean_hist": True, "explore": 0.2})}
jobs = []
for ds in ["elliptic", "amlsim10k"]:
    for s in [0, 1]:
        for v, (fl, o) in V.items():
            jobs.append((f"{ds}__dev__realistic+{v}__s{s}", ["--dataset", ds, "--split", "dev", "--scenario", "realistic", "--seed", str(s),
                         "--baselines", "none", "--suffix", f"+{v}", "--amx", json.dumps(o)] + fl))
with ThreadPoolExecutor(2) as ex:
    for tag, st in ex.map(lambda j: run(j, 5), jobs):
        print(time.strftime("%H:%M:%S"), tag, st, flush=True)
