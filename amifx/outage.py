"""Evidence-outage robustness (fault tolerance of the modular design).

At a test step t one evidence source becomes unavailable at inference time for every case
(e.g. the address/wallet service, the graph store or the counterparty aggregates are down).
  * AMIF-X: agents that read the missing view are skipped and the planner is refitted on the
    stored out-of-sample history of the remaining agents (seconds; no model retraining).
  * Monolithic LightGBM (all views / standard features): predicts with the missing features
    set to NaN (LightGBM's native missing-value handling).
  * Reference: LightGBM retrained from scratch without the missing view (full retraining).
Everything else (training data, seeds, history) is identical.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from amifx.config import AGENTS
from amifx.data.loader import Dataset, load
from amifx.evaluate import load_run, tag
from amifx.models.learners import Tabular, make_agents
from amifx.planner import Planner

DEPENDS = {"FullProfile": {"tx", "nb"}, "Transaction": {"tx"}, "Neighbourhood": {"nb"}, "Temporal": {"tp"},
           "Actor": {"ac"}, "Graph": {"tx", "gs"}, "Anomaly": {"tx"}, "NeighbourRisk": {"tx", "gs"}}
OUTAGES = ["none", "ac", "gs", "nb", "tp", "tx"]
CTX_SOURCE = {"elliptic": {"gs": ["ctx__log_deg"], "ac": ["ctx__has_wallet", "ctx__log_naddr"], "tp": ["ctx__log_volume"]},
              "amlsim10k": {"gs": ["ctx__log_deg"], "ac": ["ctx__has_history"], "tx": ["ctx__log_ncp"], "tp": ["ctx__log_volume"]}}


def run(dataset="elliptic", seed=0, t=40):
    D = load(dataset)
    df, info = load_run(tag(dataset, "main", "oracle", seed))
    lab = np.where((D.y >= 0) & (D.step < t))[0]
    cases = np.where((D.y >= 0) & (D.step == t))[0]
    y = D.y[cases].astype(int)
    A = make_agents(seed)
    for a in A.values():
        a.fit(D, lab, D.y[lab], t)
    lgb_all = Tabular("LGB-All", "all", "lgbm", seed); lgb_all.fit(D, lab, D.y[lab], t)
    lgb_std = Tabular("LGB-Std", "std", "lgbm", seed); lgb_std.fit(D, lab, D.y[lab], t)
    stp = df.step.values; yy = df.y.values.astype(int); ctx_h = D.ctx()[df.row.values]
    hist = stp <= t - 1
    rows = []
    for v in OUTAGES:
        f = D.frame.copy()
        if v != "none":
            f.loc[f.step == t, D.views[v]] = np.nan
            # context descriptors derived from the missing source are also unavailable -> history mean
            for c in CTX_SOURCE.get(dataset, {}).get(v, []):
                f.loc[f.step == t, c] = D.frame.loc[lab, c].mean()
        Dm = Dataset(D.name, f, D.src, D.dst, D.views, D.meta)
        # AMIF-X: skip affected agents, refit planner on remaining agents' stored OOS history
        keep = [a for a in AGENTS if v not in DEPENDS[a]]
        t0 = time.perf_counter()
        Ph = np.column_stack([df["c:" + a].values for a in keep])
        ok = hist & ~np.isnan(Ph).any(1)
        pl = Planner().fit(Ph[ok], yy[ok], ctx_h[ok])
        refit_s = time.perf_counter() - t0
        P = np.column_stack([A[a].predict(Dm, cases) for a in keep])
        r_amx = pl.score(P, Dm.ctx()[cases])
        r_all = lgb_all.predict(Dm, cases)
        r_std = lgb_std.predict(Dm, cases)
        row = {"seed": seed, "step": t, "outage": v, "agents_left": len(keep), "planner_refit_s": refit_s,
               "AMIF-X": 100 * ap(y, r_amx), "LightGBM-AllViews (NaN)": 100 * ap(y, r_all),
               "LightGBM (NaN)": 100 * ap(y, r_std)}
        if v != "none":
            cols = [c for c in D.views["all"] if c not in D.views[v]]
            D.views["_tmp"] = cols; D._cache.pop(("_tmp", False), None)
            t1 = time.perf_counter()
            m = Tabular("LGB-retrain", "_tmp", "lgbm", seed); m.fit(D, lab, D.y[lab], t)
            row["retrain_s"] = time.perf_counter() - t1
            row["LightGBM-AllViews retrained w/o view"] = 100 * ap(y, m.predict(D, cases))
        rows.append(row)
    return pd.DataFrame(rows)
