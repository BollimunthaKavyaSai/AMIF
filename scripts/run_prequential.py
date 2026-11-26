"""Run one (dataset, split, feedback scenario, seed) experiment.

Examples
  python scripts/run_prequential.py --dataset elliptic --scenario oracle --seed 0
  python scripts/run_prequential.py --dataset elliptic --static --seed 0
  python scripts/run_prequential.py --dataset elliptic --scenario realistic --baselines core --seed 0
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from amifx.config import PRED_DIR  # noqa: E402
from amifx.protocol import Run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="elliptic")
ap.add_argument("--split", default="main")
ap.add_argument("--scenario", default="oracle")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--baselines", default="all", help="all | core | comma-separated names | none")
ap.add_argument("--static", action="store_true")
ap.add_argument("--no-agents", action="store_true")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--cvcl", action="store_true", help="AMIF-X + cross-view consensus learning (ablation/extension)")
ap.add_argument("--suffix", default="", help="tag suffix for extra baseline-only runs, e.g. +gnn")
ap.add_argument("--sym-filter", action="store_true", help="ablation: symmetric consensus noise filter")
ap.add_argument("--amx", default="", help='AMIF-X options as JSON, e.g. {"loo": true}')
ap.add_argument("--review", default="committee", help="committee | top (ablation of disagreement-driven review)")
a = ap.parse_args()
b = a.baselines
if b not in ("all", "core"):
    b = [] if b == "none" else b.split(",")
r = Run(a.dataset, a.split, a.scenario, a.seed, baselines=b, static=a.static, agents=not a.no_agents,
        cvcl=a.cvcl, review=a.review, suffix=a.suffix, sym_filter=a.sym_filter,
        amx=__import__("json").loads(a.amx) if a.amx else None)
if a.skip_existing and (PRED_DIR / f"{r.tag}.parquet").exists():
    print("exists", r.tag)
    sys.exit(0)
r.run()
