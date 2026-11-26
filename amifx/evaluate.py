"""Evaluation of saved prequential runs: per-seed metrics, seed-pooled bootstrap CIs,
paired significance tests, and post-hoc replays of the fusion layer (ablations,
fusion variants, two-tier gating) for score-independent feedback scenarios."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from amifx import stats as st
from amifx.config import FEEDBACK, PRED_DIR, SEEDS, TIER1, AGENTS
from amifx.data.loader import load
from amifx.planner import Planner, best_f1_threshold, gate

MAIN_METRICS = ["F1", "MCC", "Precision", "Recall", "ROC-AUC", "PR-AUC", "Brier", "ECE"]


def tag(dataset, split, scenario, seed):
    return f"{dataset}__{split}__{scenario}__s{seed}"


@lru_cache(maxsize=64)
def load_run(t):
    df = pd.read_parquet(PRED_DIR / f"{t}.parquet")
    info = json.loads((PRED_DIR / f"{t}.json").read_text())
    return df, info


def available(dataset, split, scenario, seeds=SEEDS):
    return [s for s in seeds if (PRED_DIR / f"{tag(dataset, split, scenario, s)}.parquet").exists()]


def decisions(df, thr_by_step, col):
    thr = df.step.map({int(k): v for k, v in thr_by_step.items()}).values
    return (df[col].values >= thr).astype(int)


def test_part(df, info):
    sd = info["split_def"]
    return df[(df.step >= sd["test_start"]) & (df.step <= sd["last"])].reset_index(drop=True)


def group_arrays(dataset, split, scenario, group, seeds=None):
    """y (n,), scores (seeds x n), decisions (seeds x n), steps (n,) on the test cases."""
    seeds = seeds or available(dataset, split, scenario)
    S, Dd, y, stp = [], [], None, None
    for s in seeds:
        df = None
        for sx in ("", "+base", "+base2", "+gnn", "+arf"):
            if (PRED_DIR / f"{tag(dataset, split, scenario + sx, s)}.parquet").exists():
                d_, i_ = load_run(tag(dataset, split, scenario + sx, s))
                if "g:" + group in d_:
                    df, info = d_, i_
                    break
        if df is None:
            if S:
                continue
            return None
        te = test_part(df, info)
        S.append(te["g:" + group].values); Dd.append(decisions(te, info["thresholds"][group], "g:" + group))
        y, stp = te.y.values.astype(int), te.step.values
    return y, np.vstack(S), np.vstack(Dd), stp


def per_seed_metrics(y, S, Dd):
    rows = []
    for s, d in zip(S, Dd):
        m = st.metrics(y, s, 0.5); p, r, f1, mcc, acc = st._prf(y, d)
        m.update({"F1": 100 * f1, "MCC": 100 * mcc, "Precision": 100 * p, "Recall": 100 * r, "Accuracy": 100 * acc})
        rows.append(m)
    return pd.DataFrame(rows)


def pooled_boot(y, S, Dd, metric, stp=None, B=1000, block=False, seed=0, ref=None):
    """Bootstrap of the seed-averaged metric (and of its difference to `ref`)."""
    if metric in ("PR-AUC", "F1") and not block:
        return _pooled_boot_fast(y, S, Dd, metric, B, seed, ref)
    rng = np.random.default_rng(seed); f = st.metric_fn(metric)
    vals = []
    for _ in range(B):
        i = st._boot_idx(y, stp, rng, block)
        if not (0 < y[i].sum() < len(i)):
            continue
        a = np.mean([f(y[i], s[i], d[i]) for s, d in zip(S, Dd)])
        if ref is not None:
            a -= np.mean([f(y[i], s[i], d[i]) for s, d in zip(*ref)])
        vals.append(a)
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = _boot_p(vals) if ref is not None else None
    return float(lo), float(hi), p


def _boot_p(vals):
    """Two-sided bootstrap p-value. Percentile method; when every resample lies on one side of zero (so the percentile
    p-value is only bounded by 1/B), the normal approximation of the bootstrap distribution is used instead."""
    from scipy.stats import norm
    k = min((vals <= 0).mean(), (vals >= 0).mean())
    if k > 0:
        return float(min(1.0, 2 * k))
    return float(2 * norm.sf(abs(vals.mean()) / max(vals.std(ddof=1), 1e-12)))


def _pooled_boot_fast(y, S, Dd, metric, B, seed, ref):
    rng = np.random.default_rng(seed); n = len(y)
    A = [st.Presorted(y, s, d) for s, d in zip(S, Dd)]
    R = [st.Presorted(y, s, d) for s, d in zip(*ref)] if ref is not None else []
    g = (lambda P, w: P.pr_auc(w)) if metric == "PR-AUC" else (lambda P, w: P.f1(w))
    vals = []
    for _ in range(B):
        w = np.bincount(st._boot_idx(y, None, rng, False), minlength=n).astype(float)
        a = np.mean([g(P, w) for P in A])
        if R:
            a -= np.mean([g(P, w) for P in R])
        vals.append(a)
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = _boot_p(vals) if ref is not None else None
    return float(lo), float(hi), p


def summary_table(dataset, split, scenario, groups, ref="AMIF-X", B=1000, metrics_ci=("F1", "PR-AUC"), block=False):
    out = []
    R = group_arrays(dataset, split, scenario, ref) if ref else None
    for g in groups:
        A = group_arrays(dataset, split, scenario, g)
        if A is None:
            continue
        y, S, Dd, stp = A
        ps = per_seed_metrics(y, S, Dd)
        row = {"Model": g, "seeds": len(S)}
        for m in MAIN_METRICS:
            row[m] = ps[m].mean(); row[m + "_sd"] = ps[m].std(ddof=1) if len(ps) > 1 else 0.0
        for m in metrics_ci:
            lo, hi, _ = pooled_boot(y, S, Dd, m, stp, B=B, block=block)
            row[m + "_lo"], row[m + "_hi"] = lo, hi
            if R is not None and g != ref:
                dlo, dhi, p = pooled_boot(y, R[1], R[2], m, stp, B=B, block=block, ref=(S, Dd))
                row[f"d{m}_lo"], row[f"d{m}_hi"], row[f"p_{m}"] = -dhi, -dlo, p   # (g - ref)
        if R is not None and g != ref:
            # DeLong on seed-averaged scores, Wilcoxon over seed x step F1
            row["p_DeLong"] = st.delong(y, R[1].mean(0), S.mean(0))[1]
            k = min(len(R[2]), len(Dd))          # pair matching seeds only (seed lists both start at 0)
            row["p_Wilcoxon_stepF1"] = st.wilcoxon(_step_f1(y, R[2][:k], stp), _step_f1(y, Dd[:k], stp))
        out.append(row)
    T = pd.DataFrame(out)
    for m in metrics_ci:
        if f"p_{m}" in T:
            msk = T[f"p_{m}"].notna()
            T.loc[msk, f"pHolm_{m}"] = st.holm(T.loc[msk, f"p_{m}"].values)
    return T


def _step_f1(y, Dd, stp):
    v = []
    for d in Dd:
        for s in np.unique(stp):
            m = stp == s
            v.append(st._prf(y[m], d[m])[2])
    return np.array(v)


# ------------------------------------------------------------------ post-hoc replays
def replay(dataset, split, scenario, seed, members, kind="learned", context=True, interactions=False,
           tier1=None, band=None, group_for_cols=None):
    """Re-run the fusion layer prequentially on stored member scores.
    Valid for feedback scenarios whose label availability does not depend on scores."""
    fb = FEEDBACK[scenario if scenario != "static" else "oracle"]
    df, info = load_run(tag(dataset, split, scenario, seed))
    D = load(dataset)
    ctx = D.ctx()[df.row.values]
    sd = info["split_def"]
    stp = df.step.values; y = df.y.values.astype(int)
    init = stp <= sd["initial_end"]
    yobs = np.where(init, y, df.y_noisy.values)
    col = lambda c: df["c:" + c].values

    def fuse_stream(mem, ctx_on):
        P = np.column_stack([col(c) for c in mem])
        out = np.full(len(df), np.nan)
        folds = pd.Series(stp[init]).map({s: i % 3 for i, s in enumerate(np.unique(stp[init]))}).values
        idx_init = np.where(init)[0]
        for f in range(3):
            trn, tst = idx_init[folds != f], idx_init[folds == f]
            pl = Planner(kind, ctx_on, interactions=interactions).fit(P[trn], y[trn], ctx[trn] if ctx_on else None)
            out[tst] = pl.score(P[tst], ctx[tst] if ctx_on else None)
        last = None
        hist_scores = {}
        for t in range(sd["initial_end"] + 1, sd["last"] + 1):
            if scenario == "static":
                hist = np.where(init)[0]
            else:
                hist = np.where(init | (stp <= t - 1 - fb.delay))[0]
            cur = np.where(stp == t)[0]
            if scenario != "static" or last is None:
                last = Planner(kind, ctx_on, interactions=interactions).fit(P[hist], yobs[hist], ctx[hist] if ctx_on else None)
            out[cur] = last.score(P[cur], ctx[cur] if ctx_on else None)
            hist_scores[t] = (hist, last.score(P[hist], ctx[hist] if ctx_on else None))   # current planner on history
        return out, hist_scores

    r, hs = fuse_stream(members, context)
    act = None
    if tier1 is not None:
        r1, hs1 = fuse_stream(tier1, context)
        r, act = gate(r1, r, band)
        hs = {t: (hs[t][0], gate(hs1[t][1], hs[t][1], band)[0]) for t in hs}
    # same rule as the protocol for stacked models: threshold on the current planner's history scores
    thr = {t: best_f1_threshold(yobs[h], sc) for t, (h, sc) in hs.items()}
    te = (stp >= sd["test_start"]) & (stp <= sd["last"])
    d = (r >= pd.Series(stp).map(thr).values).astype(int)
    return y[te], r[te], d[te], stp[te], (act[te] if act is not None else None)


def replay_arrays(dataset, split, scenario, members, seeds=None, **kw):
    seeds = seeds or available(dataset, split, scenario)
    S, Dd, A = [], [], []
    for s in seeds:
        y, r, d, stp, act = replay(dataset, split, scenario, s, members, **kw)
        S.append(r); Dd.append(d); A.append(act)
    return y, np.vstack(S), np.vstack(Dd), stp, A
