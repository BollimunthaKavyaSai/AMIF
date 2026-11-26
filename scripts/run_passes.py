"""Wait for a running launcher (PID argv[1]) to exit, then run extra passes of jobs.py main
so that jobs that failed (e.g. killed by memory pressure) are retried. Finished jobs are skipped."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pid = int(sys.argv[1]) if len(sys.argv) > 1 else None


def alive(p):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {p}"], capture_output=True, text=True).stdout
    return str(p) in out


while pid and alive(pid):
    time.sleep(60)
env = dict(os.environ, AMIFX_WORKERS=os.environ.get("AMIFX_WORKERS", "2"), AMIFX_THREADS=os.environ.get("AMIFX_THREADS", "5"))
for k in range(2):
    with open(ROOT / "logs" / f"jobs_pass{k + 2}.log", "w") as f:
        subprocess.call([sys.executable, str(ROOT / "scripts" / "jobs.py"), "main"], stdout=f, stderr=subprocess.STDOUT,
                        env=env, cwd=ROOT)
