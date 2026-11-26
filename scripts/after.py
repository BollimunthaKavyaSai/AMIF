"""python scripts/after.py <PID> <logfile> <command...> : run a command once process PID has exited."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pid, log, cmd = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
while str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout:
    time.sleep(30)
with open(ROOT / log, "w") as f:
    subprocess.call([sys.executable] + cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
