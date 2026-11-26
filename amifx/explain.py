"""Validation of AMIF-X explanations (reviewer point 6 / editor: explanations not validated).

E1  Exactness: centred planner contributions equal brute-force Shapley values of the
    planner (2^8 coalitions, mean-imputation), i.e. they are SHAP values, not an ad-hoc score.
E2  Faithfulness with refit: removing the top-attributed agent and refitting the planner
    should lower the case risk more than removing a random agent or the agent with the
    highest raw risk ("loudest agent" heuristic).
E3  Evidence-level exact Shapley of the WHOLE pipeline over the five evidence views
    (tx, nb, tp, ac, gs; 2^5 coalitions, masked views replaced by training means), compared
    with view-grouped TreeSHAP of a monolithic LightGBM on all views (established method),
    plus a deletion test (remove the top-ranked view).
E4  Stability: agreement of case-level attributions across seeds.
No user study was run; this is stated as a limitation in the paper.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from amifx.config import AGENTS, TABLES
from amifx.data.loader import Dataset, load
from amifx.evaluate import available, load_run, replay, tag
from amifx.models.learners import Tabular, make_agents
from amifx.planner import Planner, logit, sigmoid

VIEWS = ["tx", "nb", "tp", "ac", "gs"]


def _planner_at(df, ctx, t, members):
    stp = df.step.values; y = df.y.values.astype(int)
    hist = np.where(stp <= t - 1)[0]
    P = np.column_stack([df["c:" + c].values for c in members])
    return Planner(context=True).fit(P[hist], y[hist], ctx[hist]), P


# ---------------------------------------------------------------- E1
def e1_exactness(dataset="elliptic", seed=0, t=40, n=200):
    df, info = load_run(tag(dataset, "main", "oracle", seed)); D = load(dataset)
    ctx = D.ctx()[df.row.values]
    pl, P = _planner_at(df, ctx, t, AGENTS)
    cur = np.where(df.step.values == t)[0][:n]
    phi, psi, base = pl.contributions(P[cur], ctx[cur])
    L = logit(P[cur]); mu = pl.mu_l
    beta, gamma = pl.weights()
    k = len(AGENTS)
    zc = (gamma * (ctx[cur] - pl.mu_z)).sum(1)

    def v(S):   # planner log-odds with members outside S at their history mean
        Lm = np.where(np.isin(np.arange(k), list(S))[None, :], L, mu[None, :])
        return pl.lr.intercept_[0] + (beta * Lm).sum(1) + (gamma * pl.mu_z).sum() + zc
    sh = np.zeros_like(phi)
    for j in range(k):
        rest = [i for i in range(k) if i != j]
        for r in range(k):
            for S in itertools.combinations(rest, r):
                w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                sh[:, j] += w * (v(S + (j,)) - v(S))
    full = logit(pl.score(P[cur], ctx[cur]))
    return {"max_abs_diff_shapley": float(np.abs(sh - phi).max()),
            "max_abs_additivity_error": float(np.abs(base + phi.sum(1) + psi.sum(1) - full).max()),
            "n_cases": int(len(cur)), "step": t}


# ---------------------------------------------------------------- E2
def e2_faithfulness(dataset="elliptic", seeds=None):
    seeds = seeds or available(dataset, "main", "oracle")
    rows = []
    for s in seeds:
        df, info = load_run(tag(dataset, "main", "oracle", s)); D = load(dataset)
        ctx = D.ctx()[df.row.values]; sd = info["split_def"]
        y, r_full, d_full, stp, _ = replay(dataset, "main", "oracle", s, AGENTS)
        loo = {}
        for a in AGENTS:
            loo[a] = replay(dataset, "main", "oracle", s, [m for m in AGENTS if m != a])[1]
        # centred contributions per test step
        te = np.where(df.step.values >= sd["test_start"])[0]
        PHI = np.zeros((len(te), len(AGENTS))); RAW = np.zeros_like(PHI)
        for t in sorted(np.unique(df.step.values[te])):
            pl, P = _planner_at(df, ctx, t, AGENTS)
            m = df.step.values[te] == t
            PHI[m] = pl.contributions(P[te[m]], ctx[te[m]])[0]; RAW[m] = P[te[m]]
        flagged = d_full == 1
        rng = np.random.default_rng(s)
        drops = {}
        choice = {"top-attributed (AMIF-X)": PHI.argmax(1), "highest raw risk": RAW.argmax(1),
                  "random agent": rng.integers(0, len(AGENTS), len(te)), "lowest-attributed": PHI.argmin(1)}
        LOO = np.column_stack([loo[a] for a in AGENTS])
        for nm, j in choice.items():
            r_wo = LOO[np.arange(len(te)), j]
            drops[nm] = (r_full - r_wo)[flagged]
            rows.append({"seed": s, "ranking": nm, "mean_risk_drop": float(drops[nm].mean()),
                         "mean_logodds_drop": float((logit(r_full) - logit(r_wo))[flagged].mean()),
                         "n_flagged": int(flagged.sum())})
    T = pd.DataFrame(rows)
    return T.groupby("ranking")[["mean_risk_drop", "mean_logodds_drop"]].agg(["mean", "std"]).round(4), T


# ---------------------------------------------------------------- E3
def _substep(D: Dataset, t):
    """Sub-dataset with the nodes of step t only (the graph is within-step, so exact)."""
    rows = np.where(D.step == t)[0]
    pos = -np.ones(D.n, dtype=np.int64); pos[rows] = np.arange(len(rows))
    m = (D.step[D.src] == t)
    sub = Dataset(D.name, D.frame.iloc[rows].reset_index(drop=True), pos[D.src[m]], pos[D.dst[m]], D.views, D.meta)
    return sub, rows


def e3_view_shapley(dataset="elliptic", seed=0, t=40, max_cases=600):
    D = load(dataset)
    df, info = load_run(tag(dataset, "main", "oracle", seed))
    lab = np.where((D.y >= 0) & (D.step < t))[0]
    agents = make_agents(seed)
    for a in agents.values():
        a.fit(D, lab, D.y[lab], t)
    lgb = Tabular("LGB-All", "all", "lgbm", seed); lgb.fit(D, lab, D.y[lab], t)
    ctx_all = D.ctx()[df.row.values]
    pl, _ = _planner_at(df, ctx_all, t, AGENTS)
    sub, rows = _substep(D, t)
    cases = np.where(sub.y >= 0)[0]
    if len(cases) > max_cases:
        cases = np.sort(np.random.default_rng(seed).choice(cases, max_cases, replace=False))
    means = D.frame.iloc[lab].mean(numeric_only=True)
    base_frame = sub.frame.copy()
    vals = {}
    for r in range(len(VIEWS) + 1):
        for S in itertools.combinations(VIEWS, r):
            f = base_frame.copy()
            for v in VIEWS:
                if v not in S:
                    cols = D.views[v]
                    f[cols] = np.broadcast_to(means[cols].values.astype("float32"), (len(f), len(cols)))
            sd = Dataset(D.name, f, sub.src, sub.dst, D.views, D.meta)
            P = np.column_stack([agents[a].predict(sd, cases) for a in AGENTS])
            vals[S] = logit(pl.score(P, sd.ctx()[cases]))
    k = len(VIEWS); SH = np.zeros((len(cases), k))
    for j, v in enumerate(VIEWS):
        rest = [x for x in VIEWS if x != v]
        for r in range(k):
            for S in itertools.combinations(rest, r):
                w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                S_with = tuple(x for x in VIEWS if x in S or x == v)
                SH[:, j] += w * (vals[S_with] - vals[tuple(x for x in VIEWS if x in S)])
    # monolithic LightGBM TreeSHAP grouped by view
    Xa = sub.X("all")[cases]
    contrib = lgb.m_model.booster_.predict(Xa, pred_contrib=True)[:, :-1]
    colview = np.array([c.split("__")[0] for c in D.views["all"]])
    pref = {"tx": "tx", "nb": "nb", "tp": "tp", "ac": "ac", "gs": "gs"}
    TS = np.column_stack([contrib[:, colview == pref[v]].sum(1) for v in VIEWS])
    full = vals[tuple(VIEWS)]
    y = sub.y[cases].astype(int)
    rng = np.random.default_rng(seed)
    out = {"seed": seed, "step": t, "n_cases": int(len(cases)),
           "mean_abs_shapley_by_view": dict(zip(VIEWS, np.abs(SH).mean(0).round(4).tolist())),
           "mean_abs_treeshap_by_view": dict(zip(VIEWS, np.abs(TS).mean(0).round(4).tolist())),
           "top_view_agreement": float((SH.argmax(1) == TS.argmax(1)).mean()),
           "spearman_mean": float(np.nanmean([spearmanr(a, b).correlation for a, b in zip(SH, TS)])),
           "illicit_top_view_agreement": float((SH.argmax(1) == TS.argmax(1))[y == 1].mean()) if y.sum() else None}
    # deletion: drop one view from AMIF-X, chosen by each ranking, measured on illicit cases
    for nm, j in {"AMIF-X view Shapley": SH.argmax(1), "LightGBM TreeSHAP": TS.argmax(1),
                  "random view": rng.integers(0, k, len(cases))}.items():
        wo = np.array([vals[tuple(x for x in VIEWS if x != VIEWS[jj])][i] for i, jj in enumerate(j)])
        out[f"deletion_logodds_drop[{nm}]"] = float((full - wo)[y == 1].mean()) if y.sum() else None
        out[f"deletion_risk_drop[{nm}]"] = float((sigmoid(full) - sigmoid(wo))[y == 1].mean()) if y.sum() else None
    return out, SH, TS, cases, rows


# ---------------------------------------------------------------- E4
def e4_stability(dataset="elliptic", seeds=None):
    seeds = seeds or available(dataset, "main", "oracle")
    PH = []
    for s in seeds:
        df, info = load_run(tag(dataset, "main", "oracle", s)); D = load(dataset)
        ctx = D.ctx()[df.row.values]; sd = info["split_def"]
        te = np.where(df.step.values >= sd["test_start"])[0]
        PHI = np.zeros((len(te), len(AGENTS)))
        for t in sorted(np.unique(df.step.values[te])):
            pl, P = _planner_at(df, ctx, t, AGENTS)
            m = df.step.values[te] == t
            PHI[m] = pl.contributions(P[te[m]], ctx[te[m]])[0]
        PH.append(PHI)
        yte = df.y.values[te]
    rho, top = [], []
    for a, b in itertools.combinations(range(len(PH)), 2):
        rho.append(np.nanmean([spearmanr(x, z).correlation for x, z in zip(PH[a], PH[b])]))
        top.append((PH[a].argmax(1) == PH[b].argmax(1)).mean())
    return {"pairwise_case_spearman_mean": float(np.mean(rho)), "pairwise_top1_agreement": float(np.mean(top)),
            "illicit_mean_abs_phi": dict(zip(AGENTS, np.mean([np.abs(p[yte == 1]).mean(0) for p in PH], 0).round(4).tolist())),
            "licit_mean_abs_phi": dict(zip(AGENTS, np.mean([np.abs(p[yte == 0]).mean(0) for p in PH], 0).round(4).tolist()))}
