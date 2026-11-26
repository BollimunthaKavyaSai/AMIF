"""Analysis steps that replay the planner (need the data set in memory): ablations and gating."""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import analyse as A
from amifx.config import RESULTS
p = RESULTS / "numbers.json"
if p.exists():
    A.NUM.update(json.loads(p.read_text()))
for ds in sys.argv[1:]:
    A.ablations(ds); A.gating(ds)
    p.write_text(json.dumps(A.NUM, indent=1, default=float)); print("done", ds, flush=True)
