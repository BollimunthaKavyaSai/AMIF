"""Central configuration: paths, temporal splits, seeds, feedback scenarios.

Every experiment in the paper is defined here so that a single place documents the
protocol. Nothing in this file depends on labels of the test period.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RAW_ELLIPTIC = Path(os.environ.get("AMIFX_ELLIPTIC", PROJECT / "AIMF_Project" / "primary_dataset"))
RAW_AMLSIM = {
    "amlsim10k": Path(os.environ.get("AMIFX_AMLSIM10K", PROJECT / "AIMF_Project" / "10Kvertices-1Medges")),
    "amlsim100k": Path(os.environ.get("AMIFX_AMLSIM100K", PROJECT / "AIMF_Project" / "100Kvertices-10Medges")),
}
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _d in (CACHE, RESULTS, PRED_DIR, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_THREADS = int(os.environ.get("AMIFX_THREADS", max(1, (os.cpu_count() or 4) // 3)))


@dataclass(frozen=True)
class Split:
    """Temporal protocol for one dataset.

    initial_end : last step whose labels form the initial (historical) training set.
    test_start  : first evaluated step. Steps initial_end+1 .. test_start-1 are the
                  prequential warm-up used to give the planner and every baseline an
                  out-of-sample history for threshold / fusion fitting.
    last        : last step of the stream.
    """
    name: str
    initial_end: int
    test_start: int
    last: int

    @property
    def stream(self):
        return list(range(self.initial_end + 1, self.last + 1))

    @property
    def test_steps(self):
        return list(range(self.test_start, self.last + 1))


SPLITS = {
    # main split = conference / journal split (train 1-30, val 31-35, test 36-49)
    "elliptic": {
        "main": Split("main", 30, 36, 49),
        "early": Split("early", 22, 28, 49),
        "late": Split("late", 36, 42, 49),
        "dev": Split("dev", 20, 25, 35),      # method development only (ends before test steps 36-49)
    },
    # AMLSim: 200 days -> 20 windows of 10 days
    "amlsim10k": {
        "main": Split("main", 8, 11, 20),
        "early": Split("early", 6, 9, 20),
        "late": Split("late", 11, 14, 20),
        "dev": Split("dev", 5, 7, 10),        # method development only (ends before test windows 11-20)
    },
    "amlsim100k": {
        "main": Split("main", 8, 11, 20),
    },
}
# Elliptic: dark-market shutdown happens at step 43 (Weber et al., 2019)
SHUTDOWN_STEP = 43


@dataclass(frozen=True)
class Feedback:
    """How analyst feedback reaches the learner.

    delay        : a case investigated at step s has its label available from step s+1+delay.
    budget       : fraction of each step's cases reviewed by analysts (top risk by the
                   learner's own score). None = every case is reviewed (oracle).
    audit        : extra fraction reviewed uniformly at random (exploration / audit sample).
    fn_rate      : P(analyst records an illicit case as licit) among reviewed cases.
    fp_rate      : P(analyst records a licit case as illicit) among reviewed cases.
    """
    name: str
    delay: int = 0
    budget: float | None = None
    audit: float = 0.0
    fn_rate: float = 0.0
    fp_rate: float = 0.0

    @property
    def is_oracle(self):
        return self.delay == 0 and self.budget is None and self.fn_rate == 0 and self.fp_rate == 0


FEEDBACK = {f.name: f for f in [
    Feedback("oracle"),
    Feedback("delay1", delay=1),
    Feedback("delay2", delay=2),
    Feedback("delay4", delay=4),
    Feedback("budget20", budget=0.20, audit=0.01),
    Feedback("budget10", budget=0.10, audit=0.01),
    Feedback("budget5", budget=0.05, audit=0.01),
    Feedback("noise10", fn_rate=0.10, fp_rate=0.02),
    Feedback("noise20", fn_rate=0.20, fp_rate=0.05),
    Feedback("noise30", fn_rate=0.30, fp_rate=0.10),
    # realistic combination: two-step delay, 10% review budget + 1% random audit, noisy analysts
    Feedback("realistic", delay=2, budget=0.10, audit=0.01, fn_rate=0.10, fp_rate=0.02),
    Feedback("none"),  # placeholder name for the static (no-feedback) protocol
]}

AGENTS = ["FullProfile", "Transaction", "Neighbourhood", "Temporal", "Actor", "Graph", "Anomaly", "NeighbourRisk"]
TIER1 = ["FullProfile", "Transaction", "Temporal"]
TIER2 = [a for a in AGENTS if a not in TIER1]
