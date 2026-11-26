"""Metrics and statistical tests (reviewer point 5).

* point metrics: F1 / precision / recall / MCC at the model's own history-tuned threshold,
  ROC-AUC, PR-AUC (average precision), Brier, ECE;
* 95% confidence intervals: stratified case bootstrap and, as a sensitivity analysis,
  a block bootstrap over time steps (cases inside a step are not independent);
* paired comparisons on the same test cases: paired bootstrap of the metric difference
  (two-sided p-value), DeLong's test for ROC-AUC, McNemar's exact test for decisions;
* across seeds: mean +- s.d., and a Wilcoxon signed-rank test over (seed, step) pairs;
* Holm-Bonferroni correction across the family of baseline comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sst
from sklearn.metrics import average_precision_score, roc_auc_score


def _prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return p, r, f1, mcc, (tp + tn) / max(len(y), 1)


def ece(y, p, bins=15):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, s, thr):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    pred = (s >= thr).astype(int)
    p, r, f1, mcc, acc = _prf(y, pred)
    out = {"Precision": 100 * p, "Recall": 100 * r, "F1": 100 * f1, "MCC": 100 * mcc, "Accuracy": 100 * acc}
    if 0 < y.sum() < len(y):
        out["ROC-AUC"] = 100 * roc_auc_score(y, s)
        out["PR-AUC"] = 100 * average_precision_score(y, s)
    else:
        out["ROC-AUC"] = out["PR-AUC"] = np.nan
    out["Brier"] = float(np.mean((s - y) ** 2))
    out["ECE"] = ece(y, s)
    return out


def metric_fn(name):
    if name == "PR-AUC":
        return lambda y, s, d: 100 * average_precision_score(y, s)
    if name == "ROC-AUC":
        return lambda y, s, d: 100 * roc_auc_score(y, s)
    if name in ("F1", "MCC", "Precision", "Recall"):
        def f(y, s, d):
            p, r, f1, mcc, _ = _prf(y, d)
            return 100 * {"F1": f1, "MCC": mcc, "Precision": p, "Recall": r}[name]
        return f
    raise ValueError(name)


def _boot_idx(y, groups, rng, block):
    if block:
        ug = np.unique(groups)
        pick = rng.choice(ug, len(ug), replace=True)
        return np.concatenate([np.where(groups == g)[0] for g in pick])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, s, d, metric, groups=None, B=1000, block=False, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); f = metric_fn(metric)
    vals = []
    for _ in range(B):
        i = _boot_idx(y, groups, rng, block)
        if 0 < y[i].sum() < len(i):
            vals.append(f(y[i], s[i], d[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y, sa, da, sb, db, metric, groups=None, B=2000, block=False, seed=0):
    """Difference metric(A) - metric(B) on identical cases: estimate, 95% CI, two-sided p."""
    rng = np.random.default_rng(seed); f = metric_fn(metric)
    obs = f(y, sa, da) - f(y, sb, db)
    diffs = []
    for _ in range(B):
        i = _boot_idx(y, groups, rng, block)
        if 0 < y[i].sum() < len(i):
            diffs.append(f(y[i], sa[i], da[i]) - f(y[i], sb[i], db[i]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p-value: proportion of the centred bootstrap distribution at least as extreme as observed
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) + 1.0 / len(diffs)))
    return float(obs), float(lo), float(hi), p


# ---------------------------------------------------------------- DeLong (fast, Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, sa, sb):
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    preds = np.vstack([sa, sb])
    tx = np.empty((2, m)); ty = np.empty((2, n)); tz = np.empty((2, m + n))
    for r in range(2):
        tx[r] = _midrank(preds[r, pos]); ty[r] = _midrank(preds[r, neg])
        tz[r] = _midrank(np.concatenate([preds[r, pos], preds[r, neg]]))
    auc = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(max(var, 1e-12))
    return float(auc[0] - auc[1]), float(2 * sst.norm.sf(abs(z)))


def mcnemar(y, da, db):
    a_ok = da == y; b_ok = db == y
    n01 = int((a_ok & ~b_ok).sum()); n10 = int((~a_ok & b_ok).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(sst.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(sst.wilcoxon(a, b, zero_method="zsplit").pvalue)


def topk(y, s, frac):
    n = len(y); k = max(1, int(round(frac * n))); idx = np.argsort(-s, kind="stable")[:k]
    tp = int(y[idx].sum()); nf = int(y.sum())
    return {"Budget": f"{frac * 100:g}%", "K": k, "TP@K": tp, "Precision@K": 100 * tp / k,
            "Recall@K": 100 * tp / max(nf, 1), "Lift@K": (tp / k) / (nf / n)}


# ---------------------------------------------------------------- fast weighted metrics for the bootstrap
class Presorted:
    """Scores sorted once; bootstrap replicates are evaluated from resampling counts in O(n)."""

    def __init__(self, y, s, d):
        o = np.argsort(-np.asarray(s), kind="stable")
        self.o, self.y, self.d = o, np.asarray(y)[o].astype(float), np.asarray(d)[o].astype(float)
        ss = np.asarray(s)[o]
        self.last = np.r_[ss[1:] != ss[:-1], True]          # end of each block of tied scores

    def pr_auc(self, w):
        w = w[self.o]; tp = np.cumsum(w * self.y); al = np.cumsum(w)
        # precision evaluated at the end of each tie block (as in average_precision_score)
        idx = np.where(self.last)[0]
        tp_b, al_b = tp[idx], al[idx]
        dtp = np.diff(np.r_[0.0, tp_b])
        P = tp[-1]
        prec = np.divide(tp_b, al_b, out=np.zeros_like(tp_b), where=al_b > 0)   # undrawn top ranks contribute 0
        return 100 * float((dtp * prec).sum() / P) if P > 0 else np.nan

    def f1(self, w):
        w = w[self.o]; tp = (w * self.y * self.d).sum(); fp = (w * (1 - self.y) * self.d).sum()
        fn = (w * self.y * (1 - self.d)).sum()
        return 100 * 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
