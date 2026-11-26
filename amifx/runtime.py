"""Measured wall-clock cost of full vs. two-tier gated inference (editor: "the efficiency
claim counts agent calls, not measured running time").

For a test step t, the agents are fitted on the history, then for the cases of step t we
time (a) all eight agents + full planner on every case and (b) Tier-1 agents + Tier-1
planner on every case, and Tier-2 agents + full planner only on the activated cases.
Each timing builds a fresh per-step dataset, so graph propagation and neighbour-risk
aggregation are paid inside the measurement. Median of `reps` repetitions.
"""
from __future__ import annotations

import time

import numpy as np

from amifx.config import AGENTS, TIER1, TIER2
from amifx.data.loader import Dataset, load
from amifx.evaluate import load_run, tag
from amifx.explain import _substep
from amifx.models.learners import make_agents
from amifx.planner import Planner


def measure(dataset="elliptic", seed=0, t=40, bands=((0.1, 0.8), (0.05, 0.9), (0.02, 0.95)), reps=5):
    D = load(dataset)
    df, info = load_run(tag(dataset, "main", "oracle", seed))
    lab = np.where((D.y >= 0) & (D.step < t))[0]
    A = make_agents(seed)
    fit_t = {}
    for k, a in A.items():
        t0 = time.perf_counter(); a.fit(D, lab, D.y[lab], t); fit_t[k] = time.perf_counter() - t0
    ctx = D.ctx()[df.row.values]; stp = df.step.values; y = df.y.values.astype(int)
    hist = np.where(stp <= t - 1)[0]
    Pf = np.column_stack([df["c:" + c].values for c in AGENTS]); P1 = np.column_stack([df["c:" + c].values for c in TIER1])
    full_pl = Planner().fit(Pf[hist], y[hist], ctx[hist]); t1_pl = Planner().fit(P1[hist], y[hist], ctx[hist])
    sub0, rows = _substep(D, t)

    def fresh():
        return Dataset(D.name, sub0.frame, sub0.src, sub0.dst, D.views, D.meta)
    cases = np.where(sub0.y >= 0)[0]
    res = {"step": t, "n_cases": int(len(cases)), "fit_seconds": fit_t}
    per_agent = {k: [] for k in AGENTS}
    tf = []
    for _ in range(reps):
        sd = fresh(); t0 = time.perf_counter()
        cols = []
        for k in AGENTS:
            a0 = time.perf_counter(); cols.append(A[k].predict(sd, cases)); per_agent[k].append(time.perf_counter() - a0)
        full_pl.score(np.column_stack(cols), sd.ctx()[cases]); tf.append(time.perf_counter() - t0)
    res["full_seconds"] = float(np.median(tf))
    res["agent_seconds"] = {k: float(np.median(v)) for k, v in per_agent.items()}
    for band in bands:
        tg, rate = [], None
        for _ in range(reps):
            sd = fresh(); t0 = time.perf_counter()
            p1 = np.column_stack([A[k].predict(sd, cases) for k in TIER1])
            r1 = t1_pl.score(p1, sd.ctx()[cases])
            act = cases[(r1 >= band[0]) & (r1 <= band[1])]
            if len(act):
                p2 = {k: A[k].predict(sd, act) for k in TIER2}
                pos = np.searchsorted(cases, act)
                P = np.column_stack([p1[pos, TIER1.index(k)] if k in TIER1 else p2[k] for k in AGENTS])
                full_pl.score(P, sd.ctx()[act])
            tg.append(time.perf_counter() - t0); rate = len(act) / len(cases)
        res[f"gated{band}"] = {"seconds": float(np.median(tg)), "activation_rate": rate,
                               "speedup": res["full_seconds"] / float(np.median(tg))}
    return res
