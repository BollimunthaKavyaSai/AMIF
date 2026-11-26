"""Meta-planner (fusion), decision threshold and two-tier gate.

The planner is, honestly named, a stacked-generalisation layer (Wolpert, 1992): an
L2-regularised logistic regression over the log-odds of the member models, optionally
with case-context features. It is fitted only on *out-of-sample* member predictions of
cases whose (observed) labels were available at fitting time.

Explanations. For a linear-logistic planner the contribution of member j to the
log-odds of case i, measured against the average history case,

        phi_ij = beta_j * (logit r_ij - E_hist[logit r_j]),

is exactly the interventional Shapley value of member j (Lundberg & Lee, 2017, linear
model case). Contributions + baseline log-odds sum exactly to the case's log-odds.
The earlier manuscript reported un-centred terms beta_j*logit(r_ij) and a raw context
term gamma^T z, which is why a "context" term of +1.21 appeared to dominate. Here the
context block is centred too (z is standardised on the history, so E[z]=0).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-5


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Planner:
    def __init__(self, kind="learned", context=True, C=1.0, interactions=False):
        self.kind, self.context, self.C, self.interactions = kind, context, C, interactions

    def _design(self, P, Z):
        L = logit(P)
        L = (L - self.mu_l) / self.sd_l
        if not self.context or Z is None:
            return L
        Zs = (Z - self.mu_z) / self.sd_z
        if self.interactions:
            inter = np.einsum("ij,ik->ijk", L, Zs).reshape(len(L), -1)
            return np.hstack([L, Zs, inter])
        return np.hstack([L, Zs])

    def fit(self, P, y, Z=None, w=None):
        L = logit(P)
        self.mu_l, self.sd_l = L.mean(0), L.std(0) + 1e-6
        if Z is not None:
            self.mu_z, self.sd_z = Z.mean(0), Z.std(0) + 1e-6
        if self.kind == "anchor":
            return self._fit_anchor(P, y, Z, w)
        if self.kind != "learned":
            return self
        if len(np.unique(y)) < 2:
            self.kind_fallback = True
            return self
        self.kind_fallback = False
        self.lr = LogisticRegression(C=self.C, max_iter=1000).fit(self._design(P, Z), y, sample_weight=w)
        return self

    # --- anchored planner: offset = log-odds of the anchor member (column 0), shrunken corrections
    def _fit_anchor(self, P, y, Z, w, lam=0.1):
        from scipy.optimize import minimize
        self.kind_fallback = len(np.unique(y)) < 2
        X = self._design(P, Z)[:, 1:]; off = logit(P[:, 0]); n, k = X.shape
        w = np.ones(n) if w is None else w / w.mean()
        def f(b):
            z = off + X @ b[:k] + b[k]; p = sigmoid(z)
            ll = -(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).mean()
            g = X.T @ (w * (p - y)) / n; gb = (w * (p - y)).mean()
            return ll + lam * (b[:k] ** 2).sum() / 2, np.r_[g + lam * b[:k], gb]
        self.b = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B").x if not self.kind_fallback else np.zeros(k + 1)
        class _LR:   # minimal stand-in so weights()/contributions() keep working
            pass
        self.lr = _LR(); k_l = P.shape[1]
        coef = np.r_[self.sd_l[0], self.b[:k]]          # anchor enters with unit weight on the raw log-odds scale
        self.lr.coef_ = coef[None, :]; self.lr.intercept_ = np.array([self.b[k] + self.mu_l[0]])
        return self

    # --- scoring
    def score(self, P, Z=None):
        if self.kind == "anchor":
            return sigmoid(logit(P[:, 0]) + self._design(P, Z)[:, 1:] @ self.b[:-1] + self.b[-1])
        if self.kind == "equal" or (getattr(self, "kind_fallback", False) and self.kind != "anchor"):  # AMIF-style fixed fusion
            return sigmoid(logit(P).mean(1))
        if self.kind == "mean_prob":
            return P.mean(1)
        if self.kind == "confidence":                                         # |p - 0.5| weighted average
            w = np.abs(P - 0.5) + 1e-3
            return (w * P).sum(1) / w.sum(1)
        return self.lr.predict_proba(self._design(P, Z))[:, 1]

    # --- explanation (exact for the linear planner without interactions)
    def weights(self):
        """Per-member coefficient on the raw log-odds scale and context coefficients."""
        k = len(self.mu_l)
        c = self.lr.coef_[0]
        beta = c[:k] / self.sd_l
        gamma = c[k:k + len(getattr(self, "mu_z", []))] / getattr(self, "sd_z", 1.0) if self.context else np.array([])
        return beta, gamma

    def contributions(self, P, Z=None):
        """Centred contributions (n x k members, n x m context) and the baseline log-odds."""
        beta, gamma = self.weights()
        L = logit(P)
        phi = beta * (L - self.mu_l)
        if self.context and Z is not None and not self.interactions:
            psi = gamma * (Z - self.mu_z)
        else:
            psi = np.zeros((len(P), 0))
        base = self.lr.intercept_[0]          # the design is centred on the history, so the baseline is the intercept
        return phi, psi, float(base)


def best_f1_threshold(y, s):
    """Exact F1-maximising threshold on (y, s); ties broken towards the larger threshold."""
    y = np.asarray(y); s = np.asarray(s)
    if len(y) == 0 or y.sum() == 0:
        return 0.5
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    last = np.r_[ss[1:] != ss[:-1], True]           # only cut between distinct scores
    f1 = np.where(last, f1, -1)
    i = int(np.argmax(f1))
    return float(ss[i])


def gate(r1, r_full, band):
    """Two-tier gate: Tier-2 agents are run only when the Tier-1 risk lies inside `band`."""
    lo, hi = band
    act = (r1 >= lo) & (r1 <= hi)
    return np.where(act, r_full, r1), act
