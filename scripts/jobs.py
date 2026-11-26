"""Final experiment grid (frozen AMIF-X configuration, chosen on the DEV split; see scripts/dev*.py).

  python scripts/jobs.py main        # P1 -> P2 -> P3, resumable (finished runs are skipped)
Each experiment is split into a baseline run (tag suffix +base, no agents), an AMIF-X run (main tag)
and, where relevant, the CVCL extension (+cvcl). Memory: 7.7 GB RAM -> at most two concurrent jobs.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"; LOGS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
B_ORACLE = ("LR,RF,XGBoost,LightGBM,CatBoost,MLP,LightGBM-AllViews,LR-AllViews,XGBoost-AllViews,SGC,"
            "LightGBM-Window,LightGBM-Decay,LightGBM-DDM,LightGBM-Collective")
B_FEEDBACK = "LR,RF,XGBoost,LightGBM,CatBoost,LightGBM-AllViews,LightGBM-Decay,LightGBM-SelfTrain,LightGBM-Collective"
B_STATIC = "LR,RF,XGBoost,LightGBM,CatBoost,MLP,LightGBM-AllViews,LR-AllViews,XGBoost-AllViews,SGC,LightGBM-Collective"
B_SPLIT = "LR,RF,XGBoost,LightGBM,CatBoost,LightGBM-AllViews,LightGBM-Collective"
CVCL = json.dumps({"clean_hist": True, "explore": 0.2})          # CVCL extension = dev variant v7


def J(ds, scen, seed, split="main", baselines="none", agents=True, suffix="", static=False, extra=()):
    tag = f"{ds}__{split}__{'static' if static else scen}{suffix}__s{seed}"
    a = ["--dataset", ds, "--split", split, "--scenario", scen, "--seed", str(seed), "--baselines", baselines] + list(extra)
    if suffix:
        a += ["--suffix", suffix]
    if not agents:
        a.append("--no-agents")
    if static:
        a.append("--static")
    return tag, a


def base(ds, scen, s, b, split="main"):
    """Baseline run. If no complete '+base' result exists yet, the baselines are split over two
    processes ('+base' and '+base2') so that each fits in the 7.7 GB of RAM."""
    if (ROOT / "results" / "predictions" / f"{ds}__{split}__{scen}+base__s{s}.parquet").exists():
        return [J(ds, scen, s, split, b, agents=False, suffix="+base")]
    names = b.split(","); h = (len(names) + 1) // 2
    # keep Stacking members together in the first half (LR, RF, XGBoost, LightGBM, CatBoost come first)
    first = [n for n in names if n in ("LR", "RF", "XGBoost", "LightGBM", "CatBoost")]
    rest = [n for n in names if n not in first]
    return [J(ds, scen, s, split, ",".join(first), agents=False, suffix="+base"),
            J(ds, scen, s, split, ",".join(rest), agents=False, suffix="+base2")]


def amifx(ds, scen, s, split="main"):
    return J(ds, scen, s, split)


def cvcl(ds, scen, s, extra=(), suffix="+cvcl"):
    tag, a = J(ds, scen, s, suffix=suffix, extra=("--cvcl", "--amx", CVCL) + tuple(extra))
    if "--sym-filter" in extra:                       # protocol inserts -symfilter before the suffix
        tag = tag.replace(suffix + "__s", "-symfilter" + suffix + "__s")
    return tag, a


def gnn(ds, s, which):
    return J(ds, "oracle", s, baselines=which, agents=False, suffix="+gnn")


DS = ["elliptic", "amlsim10k"]
GROUPS = {"P1": [], "P2": [], "P3": []}
for ds in DS:
    GROUPS["P1"] += [amifx(ds, sc, s) for sc in ["realistic", "oracle"] for s in SEEDS]
    GROUPS["P1"] += [cvcl(ds, "realistic", s) for s in SEEDS]
    GROUPS["P1"] += [J(ds, "oracle", s, baselines=B_STATIC, static=True) for s in SEEDS]
# headline baselines: realistic on both data sets and Elliptic++ oracle with 5 seeds; AMLSim oracle with 3
# recover a half that was interrupted by a restart (+base2 = baselines not run in the +base process)
_first = {"LR", "RF", "XGBoost", "LightGBM", "CatBoost"}
GROUPS["P1"] += [J("elliptic", "oracle", 4, "main", ",".join(n for n in B_ORACLE.split(",") if n not in _first), agents=False, suffix="+base2")]
GROUPS["P1"] += [j for ds in DS for s in SEEDS for j in base(ds, "realistic", s, B_FEEDBACK)]
GROUPS["P1"] += [j for s in SEEDS for j in base("elliptic", "oracle", s, B_ORACLE)]
GROUPS["P1"] += [j for s in SEEDS[:3] for j in base("amlsim10k", "oracle", s, B_ORACLE)]
GROUPS["P1"] += [gnn("elliptic", s, "GCN,GraphSAGE,GAT") for s in SEEDS[:3]]
GROUPS["P1"] += [gnn("amlsim10k", s, "GCN,GraphSAGE") for s in SEEDS[:2]]
# robustness (2 seeds), ordered by importance: alternative temporal splits (realistic, then oracle), the
# symmetric-filter ablation, then other feedback regimes vs. the strongest baselines
SWEEP_B = "LightGBM,LightGBM-AllViews,LightGBM-Collective"
for sc in ["realistic", "oracle"]:
    for ds in DS:
        for sp in ["early", "late"]:
            GROUPS["P2"] += [J(ds, sc, s, split=sp, baselines=SWEEP_B) for s in SEEDS[:2]]
for ds in DS:
    GROUPS["P2"] += [cvcl(ds, "realistic", 0, extra=("--sym-filter",))]
for sc in ["budget10", "noise20", "delay2"]:
    for ds in DS:
        GROUPS["P2"] += [amifx(ds, sc, s) for s in SEEDS[:2]]
        GROUPS["P2"] += [J(ds, sc, s, "main", SWEEP_B, agents=False, suffix="+base") for s in SEEDS[:2]]
ORDER = ["P1", "P2", "P3"]


def run(job, threads):
    tag, args = job
    if (ROOT / "results" / "predictions" / f"{tag}.parquet").exists():
        return tag, "skip"
    env = dict(os.environ, AMIFX_THREADS=str(threads), PYTHONUNBUFFERED="1")
    t = time.time()
    with open(LOGS / f"{tag}.log", "w") as f:
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "run_prequential.py")] + args, stdout=f,
                             stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    return tag, f"rc={rc} {time.time() - t:.0f}s"


if __name__ == "__main__":
    names = ORDER if sys.argv[1] == "main" else sys.argv[1].split(",")
    workers = int(os.environ.get("AMIFX_WORKERS", 2)); threads = int(os.environ.get("AMIFX_THREADS", 5))
    jobs = [j for n in names for j in GROUPS[n]]
    print(f"{len(jobs)} jobs, {workers} workers x {threads} threads", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for tag, st in ex.map(lambda j: run(j, threads), jobs):
            print(time.strftime("%H:%M:%S"), tag, st, flush=True)
