"""Analysis steps that only read stored predictions (low memory): run while experiments continue."""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import analyse as A
from amifx.config import RESULTS
ds = sys.argv[1]
p = RESULTS / "numbers.json"
if p.exists():
    A.NUM.update(json.loads(p.read_text()))
A.main_tables(ds); A.decomposition(ds); A.periods(ds); A.budget(ds); A.per_step_fig(ds); A.pr_fig(ds); A.weights_fig(ds); A.cvcl_ablation(ds)
p.write_text(json.dumps(A.NUM, indent=1, default=float)); print("done", ds)
