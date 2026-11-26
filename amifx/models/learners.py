"""All learners (AMIF-X specialist agents and every baseline) behind one interface.

    fit(D, tr, y_tr, t)  -> train on rows `tr` with *observed* labels `y_tr` (labels as the
                            analysts recorded them; may be delayed / partial / noisy)
    predict(D, idx)      -> risk scores in [0, 1] for rows `idx`

Standardisation and every other fitted statistic is fitted on `tr` only, at each refit.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from amifx.config import N_THREADS

torch.set_num_threads(N_THREADS)


def _lgbm(seed, n=150, lr=0.1):
    return LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=31, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                          verbosity=-1, n_jobs=N_THREADS)


class Learner:
    name = "base"
    family = "base"
    streaming = False

    def __init__(self, seed=0):
        self.seed = seed

    def clone(self):
        c = copy.copy(self)
        c.__dict__ = {k: v for k, v in self.__dict__.items() if not k.startswith("m_")}
        c.__dict__ = copy.deepcopy(c.__dict__)
        return c

    def _single_class(self, y):
        if len(np.unique(y)) < 2:
            self.m_const = float(np.mean(y)) if len(y) else 0.0
            return True
        self.m_const = None
        return False


# ----------------------------------------------------------------------------- tabular
class Tabular(Learner):
    family = "tabular"

    def __init__(self, name, view, kind="lgbm", seed=0, transform=False, scale=False, sample_weight=None,
                 window=None, n=150):
        super().__init__(seed)
        self.name, self.view, self.kind, self.transform, self.scale = name, view, kind, transform, scale
        self.sample_weight, self.window, self.n = sample_weight, window, n

    def _model(self):
        s = self.seed
        if self.kind == "lgbm":
            return _lgbm(s, self.n)
        if self.kind == "xgb":
            return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", random_state=s, n_jobs=N_THREADS, eval_metric="logloss", verbosity=0)
        if self.kind == "cat":
            return CatBoostClassifier(iterations=200, depth=6, learning_rate=0.15, random_seed=s, verbose=0,
                                      thread_count=N_THREADS, allow_writing_files=False)
        if self.kind == "rf":
            return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", n_jobs=N_THREADS,
                                          random_state=s)
        if self.kind == "lr":
            return LogisticRegression(C=0.5, max_iter=400)
        raise ValueError(self.kind)

    def _X(self, D, idx):
        X = D.X(self.view, transform=self.transform or self.kind in ("lr", "rf"))[idx]
        if self.kind in ("lr",):
            X = self.m_scaler.transform(X)
        return X

    def fit(self, D, tr, y, t, w=None):
        if self.window is not None:
            keep = D.step[tr] >= t - self.window
            tr, y = tr[keep], y[keep]
            w = None if w is None else w[keep]
        if self._single_class(y):
            return
        if self.sample_weight == "decay":
            age = (t - 1 - D.step[tr]).clip(min=0)
            dw = 0.5 ** (age / 5.0)                    # half-life of five time steps
            w = dw if w is None else w * dw
        if self.kind == "lr":
            self.m_scaler = StandardScaler().fit(D.X(self.view, transform=True)[tr])
        m = self._model()
        X = self._X(D, tr)
        if w is not None:
            m.fit(X, y, sample_weight=w)
        else:
            m.fit(X, y)
        self.m_model = m

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        return self.m_model.predict_proba(self._X(D, idx))[:, 1].astype("float32")


class MLP(Learner):
    family = "neural"

    def __init__(self, name, view, seed=0, epochs=15):
        super().__init__(seed)
        self.name, self.view, self.epochs = name, view, epochs

    def fit(self, D, tr, y, t):
        if self._single_class(y):
            return
        X = D.X(self.view, transform=True)
        self.m_scaler = StandardScaler().fit(X[tr])
        Xt = torch.tensor(np.clip(self.m_scaler.transform(X[tr]), -8, 8), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.ReLU(),
                            nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(yt), generator=g)
            for b in range(0, len(yt), 1024):
                bi = perm[b:b + 1024]
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(net(Xt[bi]).squeeze(-1), yt[bi]).backward()
                opt.step()
        self.m_net = net.eval()

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        X = torch.tensor(np.clip(self.m_scaler.transform(D.X(self.view, transform=True)[idx]), -8, 8), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.m_net(X).squeeze(-1)).numpy().astype("float32")


# ----------------------------------------------------------------------------- AMIF-X specific agents
class AnomalyAgent(Learner):
    """Isolation Forest fitted on cases verified as licit; score mapped to a probability
    by logistic calibration on the training rows (a novelty signal that does not need
    previously seen illicit patterns)."""
    family = "anomaly"

    def __init__(self, name="Anomaly", view="tx", seed=0):
        super().__init__(seed)
        self.name, self.view = name, view

    def fit(self, D, tr, y, t):
        if self._single_class(y):
            return
        X = D.X(self.view, transform=True)
        rng = np.random.default_rng(self.seed)
        lic = tr[y == 0]
        if len(lic) > 50000:
            lic = rng.choice(lic, 50000, replace=False)
        self.m_iso = IsolationForest(n_estimators=100, max_samples=4096, random_state=self.seed, n_jobs=N_THREADS).fit(X[lic])
        sc = -self.m_iso.score_samples(X[tr]).reshape(-1, 1)
        self.m_cal = LogisticRegression().fit(sc, y)

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        sc = -self.m_iso.score_samples(D.X(self.view, transform=True)[idx]).reshape(-1, 1)
        return self.m_cal.predict_proba(sc)[:, 1].astype("float32")


class GraphAgent(Learner):
    """Decoupled two-hop GraphSAGE (mean aggregator, SIGN-style): an MLP on
    [x_v, mean_{N(v)} x, mean_{N2(v)} x]. Propagation is label-free and, because the
    graph is a union of per-step components, never mixes information across steps.
    The scaler is fitted on the training rows of each refit only."""
    family = "graph"

    def __init__(self, name="Graph", view="graphin", seed=0, epochs=30, model="mlp"):
        super().__init__(seed)
        self.name, self.view, self.epochs, self.model = name, view, epochs, model

    def _H(self, D):
        key = ("sageH", self.view)
        if key not in D._cache:
            x = torch.tensor(D.X(self.view, transform=True))
            A = D.norm_adj()
            with torch.no_grad():
                ax = torch.sparse.mm(A, x); a2x = torch.sparse.mm(A, ax)
            D._cache[key] = torch.cat([x, ax, a2x], 1).numpy()
        return D._cache[key]

    def fit(self, D, tr, y, t, w=None):
        if self._single_class(y):
            return
        H = self._H(D)
        if self.model == "lgbm":                       # stable tree learner on the propagated features
            self.m_lgb = _lgbm(self.seed).fit(H[tr], y, sample_weight=w)
            return
        self.m_scaler = StandardScaler().fit(H[tr])
        Ht = torch.tensor(np.clip(self.m_scaler.transform(H[tr]), -8, 8), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(H.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                            nn.Dropout(0.2), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(self.seed)
        bs = 4096
        epochs = max(3, min(self.epochs, int(np.ceil(3e5 / max(len(yt), 1)))))   # cap at ~600k samples seen
        for _ in range(epochs):
            perm = torch.randperm(len(yt), generator=g)
            for b in range(0, len(yt), bs):
                bi = perm[b:b + bs]
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(net(Ht[bi]).squeeze(-1), yt[bi]).backward()
                opt.step()
        self.m_net = net.eval()

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        if self.model == "lgbm":
            return self.m_lgb.predict_proba(self._H(D)[idx])[:, 1].astype("float32")
        X = torch.tensor(np.clip(self.m_scaler.transform(self._H(D)[idx]), -8, 8), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.m_net(X).squeeze(-1)).numpy().astype("float32")


class NeighbourRiskAgent(Learner):
    """Risk propagation: aggregates the Transaction-view risk of graph neighbours (labelled
    or not) and combines it with the case's own risk and structure. Training rows use
    out-of-fold neighbour risks (folds = groups of time steps), so the agent never sees
    in-sample scores. Labels of neighbours are never used."""
    family = "graph"

    def __init__(self, name="NeighbourRisk", seed=0, base_view="tx", own_view="gs"):
        super().__init__(seed)
        self.name, self.base_view, self.own_view = name, base_view, own_view

    def _feats(self, D, base, rows):
        i, j = D.neighbours()
        m = np.isin(i, rows)
        ii, pj = i[m], base[j[m]]
        ok = ~np.isnan(pj)
        ii, pj = ii[ok], pj[ok]
        N = D.n
        s = np.bincount(ii, weights=pj, minlength=N); c = np.bincount(ii, minlength=N)
        mx = np.full(N, -1.0); np.maximum.at(mx, ii, pj)
        mean = np.where(c > 0, s / np.maximum(c, 1), -1.0)
        frac_hi = np.bincount(ii, weights=(pj > 0.5).astype(float), minlength=N) / np.maximum(c, 1)
        own = np.nan_to_num(base, nan=-1.0)
        X = np.column_stack([mean, mx, c, frac_hi, own])[rows]
        return np.hstack([X, D.X(self.own_view)[rows]]).astype("float32")

    def fit(self, D, tr, y, t):
        if self._single_class(y):
            return
        Xtx = D.X(self.base_view)
        steps = np.unique(D.step[tr])
        fold_of_step = {s: k % 3 for k, s in enumerate(steps)}
        fold = np.array([fold_of_step[s] for s in D.step[tr]])
        base = np.full(D.n, np.nan, "float32")
        for k in range(3):
            trk = tr[fold != k]
            if len(np.unique(y[fold != k])) < 2:
                continue
            mk = _lgbm(self.seed, 100).fit(Xtx[trk], y[fold != k])
            nodes = np.where(np.isin(D.step, steps[[fold_of_step[s] == k for s in steps]]))[0]
            base[nodes] = mk.predict_proba(Xtx[nodes])[:, 1]
        self.m_base = _lgbm(self.seed, 100).fit(Xtx[tr], y)
        Xn = self._feats(D, base, tr)
        self.m_model = _lgbm(self.seed, 100).fit(Xn, y)

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        nodes = np.where(np.isin(D.step, np.unique(D.step[idx])))[0]     # all nodes of the scored steps
        base = np.full(D.n, np.nan, "float32")
        base[nodes] = self.m_base.predict_proba(D.X(self.base_view)[nodes])[:, 1]
        return self.m_model.predict_proba(self._feats(D, base, idx))[:, 1].astype("float32")


# ----------------------------------------------------------------------------- end-to-end GNN baselines
class GraphOps:
    """Pre-built sparse operators for the (undirected, per-step) transaction graph."""

    def __init__(self, D, nodes=None):
        if nodes is None:
            nodes = np.arange(D.n)
        pos = -np.ones(D.n, dtype=np.int64); pos[nodes] = np.arange(len(nodes))
        keep = (pos[D.src] >= 0) & (pos[D.dst] >= 0)
        es, ed = pos[D.src[keep]], pos[D.dst[keep]]
        n = len(nodes)
        src = np.concatenate([es, ed]); dst = np.concatenate([ed, es])
        loop = np.arange(n)
        s2, d2 = np.concatenate([src, loop]), np.concatenate([dst, loop])
        deg = np.bincount(d2, minlength=n).astype("float32")
        w = 1.0 / np.sqrt(deg[s2] * deg[d2])
        self.gcn = torch.sparse_coo_tensor(torch.tensor(np.vstack([d2, s2])), torch.tensor(w, dtype=torch.float32), (n, n)).coalesce()
        dm = np.bincount(dst, minlength=n).astype("float32")
        self.mean = torch.sparse_coo_tensor(torch.tensor(np.vstack([dst, src])), torch.tensor(1.0 / np.maximum(dm[dst], 1)),
                                            (n, n)).coalesce()
        self.src = torch.tensor(s2, dtype=torch.long); self.dst = torch.tensor(d2, dtype=torch.long); self.n = n


class GCNLayer(nn.Module):           # Kipf & Welling (2017)
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)

    def forward(self, x, g):
        return torch.sparse.mm(g.gcn, self.lin(x))


class SAGELayer(nn.Module):          # Hamilton et al. (2017), mean aggregator
    def __init__(self, i, o):
        super().__init__(); self.self_ = nn.Linear(i, o); self.nb = nn.Linear(i, o, bias=False)

    def forward(self, x, g):
        return self.self_(x) + self.nb(torch.sparse.mm(g.mean, x))


class GATLayer(nn.Module):           # Velickovic et al. (2018), 4 heads, concatenated
    def __init__(self, i, o, heads=4):
        super().__init__()
        self.h, self.c = heads, o // heads
        self.lin = nn.Linear(i, self.h * self.c, bias=False)
        self.a_s = nn.Parameter(torch.randn(self.h, self.c) * 0.1); self.a_d = nn.Parameter(torch.randn(self.h, self.c) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.h * self.c))

    def forward(self, x, g):
        z = self.lin(x).view(-1, self.h, self.c)
        e = F.leaky_relu((z * self.a_s).sum(-1)[g.src] + (z * self.a_d).sum(-1)[g.dst], 0.2)      # E x H
        mx = torch.full((g.n, self.h), -1e9).scatter_reduce(0, g.dst[:, None].expand(-1, self.h), e, "amax")
        a = torch.exp(e - mx[g.dst])
        den = torch.zeros(g.n, self.h).index_add_(0, g.dst, a)
        a = a / (den[g.dst] + 1e-16)
        out = torch.zeros(g.n, self.h, self.c).index_add_(0, g.dst, a[..., None] * z[g.src])
        return out.reshape(g.n, -1) + self.bias


class GNN(Learner):
    """Two-layer GCN / GraphSAGE / GAT trained end-to-end (transductive, full batch).
    Warm-started across refits (continual fine-tuning)."""
    family = "gnn"

    def __init__(self, name, kind, view="std", seed=0, hidden=32, first_epochs=60, refit_epochs=10):
        super().__init__(seed)
        self.name, self.kind, self.view, self.hidden = name, kind, view, hidden
        self.first_epochs, self.refit_epochs = first_epochs, refit_epochs
        self.state = None

    def _net(self, d):
        h = self.hidden
        mk = {"gcn": GCNLayer, "sage": SAGELayer, "gat": GATLayer}[self.kind]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c1, s.c2, s.out = mk(d, h), mk(h, h), nn.Linear(h, 1)

            def forward(s, x, g):
                x = F.dropout(F.relu(s.c1(x, g)), 0.3, s.training)
                x = F.dropout(F.relu(s.c2(x, g)), 0.3, s.training)
                return s.out(x).squeeze(-1)
        return Net()

    def _x(self, D):
        return torch.tensor(np.clip(self.m_scaler.transform(D.X(self.view, transform=True)), -8, 8), dtype=torch.float32)

    def fit(self, D, tr, y, t):
        if self._single_class(y):
            return
        self.m_scaler = StandardScaler().fit(D.X(self.view, transform=True)[tr])
        # the graph is a union of per-step components, so the subgraph of steps <= t is exact
        last = max(int(t), int(D.step[tr].max()))
        nodes = np.where(D.step <= last)[0]
        key = ("graphops", last)
        if key not in D._cache:
            D._cache = {k: v for k, v in D._cache.items() if not (isinstance(k, tuple) and k[0] == "graphops")}
            D._cache[key] = GraphOps(D, nodes)
        ei = D._cache[key]
        x = self._x(D)[nodes]
        pos = -np.ones(D.n, dtype=np.int64); pos[nodes] = np.arange(len(nodes))
        tr_local = pos[tr]
        torch.manual_seed(self.seed)
        net = self._net(x.shape[1])
        epochs = self.first_epochs
        if self.state is not None:
            net.load_state_dict(self.state); epochs = self.refit_epochs
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
        trt = torch.tensor(tr_local); yt = torch.tensor(y, dtype=torch.float32)
        pw = torch.tensor(np.sqrt((y == 0).sum() / max((y == 1).sum(), 1)), dtype=torch.float32)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(x, ei)[trt], yt, pos_weight=pw).backward()
            opt.step()
        net.eval()
        self.state = copy.deepcopy(net.state_dict())
        self.m_all = np.full(D.n, np.nan, "float32")
        with torch.no_grad():
            self.m_all[nodes] = torch.sigmoid(net(x, ei)).numpy().astype("float32")

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        return self.m_all[idx]

    def clone(self):
        c = super().clone(); c.state = None
        return c


class SGC(Learner):
    """Simplified Graph Convolution (Wu et al., 2019): logistic regression on S^K X with the
    symmetric-normalised adjacency with self-loops, K = 2. Propagation is label-free and
    within-step; the scaler and the classifier are fitted on training rows only."""
    family = "gnn"

    def __init__(self, name="SGC", view="std", seed=0, K=2):
        super().__init__(seed)
        self.name, self.view, self.K = name, view, K

    def _SX(self, D):
        key = ("sgc", self.view, self.K)
        if key not in D._cache:
            g = GraphOps(D)
            x = torch.tensor(D.X(self.view, transform=True))
            with torch.no_grad():
                for _ in range(self.K):
                    x = torch.sparse.mm(g.gcn, x)
            D._cache[key] = x.numpy()
        return D._cache[key]

    def fit(self, D, tr, y, t):
        if self._single_class(y):
            return
        X = self._SX(D)
        self.m_scaler = StandardScaler().fit(X[tr])
        self.m_model = LogisticRegression(C=0.5, max_iter=400).fit(self.m_scaler.transform(X[tr]), y)

    def predict(self, D, idx):
        if getattr(self, "m_const", None) is not None:
            return np.full(len(idx), self.m_const, "float32")
        return self.m_model.predict_proba(self.m_scaler.transform(self._SX(D)[idx]))[:, 1].astype("float32")


# ----------------------------------------------------------------------------- drift-aware baselines
class DriftTriggered(Learner):
    """LightGBM with DDM drift detection (Gama et al., 2004): the deployed model is kept
    fixed and is retrained only when its error rate on newly verified cases rises
    significantly (p+s >= p_min + 3 s_min); retraining then uses the cases verified since
    the warning level was first reached (at least the last five steps)."""
    family = "drift"

    def __init__(self, name="LGBM-DDM", view="std", seed=0):
        super().__init__(seed)
        self.name, self.view = name, view
        self.seen = set(); self.reset_stats(); self.warn_step = None; self.n_retrain = 0

    def reset_stats(self):
        self.n = 0; self.err = 0; self.pmin = np.inf; self.smin = np.inf; self.psmin = np.inf

    def clone(self):
        c = super().clone(); c.seen = set(); c.reset_stats(); c.warn_step = None; c.n_retrain = 0
        return c

    def fit(self, D, tr, y, t):
        X = D.X(self.view)
        new = np.array([k for k, r in enumerate(tr) if r not in self.seen], dtype=int)
        retrain = not hasattr(self, "m_model")
        if not retrain and len(new):
            order = new[np.argsort(D.step[tr[new]], kind="stable")]
            p = self.m_model.predict_proba(X[tr[order]])[:, 1]
            for e, s in zip(((p >= 0.5).astype(int) != y[order]).astype(int), D.step[tr[order]]):
                self.n += 1; self.err += e
                pi = self.err / self.n; si = np.sqrt(pi * (1 - pi) / self.n)
                if self.n >= 100 and pi + si < self.psmin:
                    self.pmin, self.smin, self.psmin = pi, si, pi + si
                if self.n >= 100 and pi + si >= self.pmin + 2 * self.smin and self.warn_step is None:
                    self.warn_step = s
                if self.n >= 100 and pi + si >= self.pmin + 3 * self.smin:
                    retrain = True
        self.seen.update(tr.tolist())
        if retrain:
            use = np.ones(len(tr), bool)
            if hasattr(self, "m_model"):
                start = min(self.warn_step if self.warn_step is not None else t - 5, t - 5)
                use = D.step[tr] >= start
            if len(np.unique(y[use])) < 2:
                use = np.ones(len(tr), bool)
            self.m_model = _lgbm(self.seed).fit(X[tr[use]], y[use])
            self.reset_stats(); self.warn_step = None; self.n_retrain += 1

    def predict(self, D, idx):
        return self.m_model.predict_proba(D.X(self.view)[idx])[:, 1].astype("float32")


class ARF(Learner):
    """Adaptive Random Forest (Gomes et al., 2017) with ADWIN drift detectors (river).
    A streaming learner: each verified case is learned once, in arrival order."""
    family = "drift"
    streaming = True

    def __init__(self, name="ARF", view="std", seed=0, n_models=10):
        super().__init__(seed)
        self.name, self.view, self.n_models = name, view, n_models
        self.seen = set(); self.model = None

    def clone(self):
        c = super().clone(); c.seen = set(); c.model = None
        return c

    def _init(self):
        from river import forest
        self.model = forest.ARFClassifier(n_models=self.n_models, seed=self.seed, max_features="sqrt", leaf_prediction="mc")

    def fit(self, D, tr, y, t):
        if self.model is None:
            self._init()
        X = D.X(self.view, transform=True)
        new = [k for k, r in enumerate(tr) if r not in self.seen]
        new.sort(key=lambda k: D.step[tr[k]])
        for k in new:
            self.model.learn_one(dict(enumerate(X[tr[k]].tolist())), int(y[k]))
            self.seen.add(int(tr[k]))

    def predict(self, D, idx):
        if self.model is None:
            return np.zeros(len(idx), "float32")
        X = D.X(self.view, transform=True)
        return np.array([self.model.predict_proba_one(dict(enumerate(X[i].tolist()))).get(1, 0.0) for i in idx],
                        dtype="float32")


# ----------------------------------------------------------------------------- registry
def make_agents(seed, full_view="std", decay=False, nr_view="all", graph_model="lgbm", collective_agent=False):
    """The eight AMIF-X specialist agents (one evidence view each)."""
    sw = "decay" if decay else None
    A = {
        "FullProfile": Tabular("FullProfile", full_view, "lgbm", seed, sample_weight=sw),
        "Transaction": Tabular("Transaction", "tx", "lgbm", seed, sample_weight=sw),
        "Neighbourhood": Tabular("Neighbourhood", "nb", "lgbm", seed, sample_weight=sw),
        "Temporal": Tabular("Temporal", "tp", "lgbm", seed, sample_weight=sw),
        "Actor": Tabular("Actor", "ac", "lgbm", seed, sample_weight=sw),
        "Graph": GraphAgent("Graph", "graphin", seed, model=graph_model),
        "Anomaly": AnomalyAgent("Anomaly", "tx", seed),
        "NeighbourRisk": NeighbourRiskAgent("NeighbourRisk", seed, base_view=nr_view, own_view="gs" if nr_view == "tx" else "all"),
    }
    if collective_agent:     # second relational agent: risk propagated from an all-view first stage
        A["CollectiveRisk"] = NeighbourRiskAgent("CollectiveRisk", seed, base_view="all", own_view="all")
    return A


def make_baselines(seed, which="all"):
    B = {
        "LR": Tabular("LR", "std", "lr", seed),
        "RF": Tabular("RF", "std", "rf", seed),
        "XGBoost": Tabular("XGBoost", "std", "xgb", seed),
        "LightGBM": Tabular("LightGBM", "std", "lgbm", seed),
        "CatBoost": Tabular("CatBoost", "std", "cat", seed),
        "MLP": MLP("MLP", "std", seed),
        "LightGBM-AllViews": Tabular("LightGBM-AllViews", "all", "lgbm", seed),
        "LR-AllViews": Tabular("LR-AllViews", "all", "lr", seed),
        "XGBoost-AllViews": Tabular("XGBoost-AllViews", "all", "xgb", seed),
        "GCN": GNN("GCN", "gcn", "std", seed),
        "GraphSAGE": GNN("GraphSAGE", "sage", "std", seed),
        "GAT": GNN("GAT", "gat", "std", seed),
        "SGC": SGC("SGC", "std", seed),
        # two-stage collective classification: first-stage LightGBM on all views, neighbour-risk aggregates
        # added to all-view features (the strongest non-modular form of AMIF-X's risk propagation)
        "LightGBM-Collective": NeighbourRiskAgent("LightGBM-Collective", seed, base_view="all", own_view="all"),
        "LightGBM-SelfTrain": Tabular("LightGBM-SelfTrain", "std", "lgbm", seed),   # pseudo-labels handled in protocol
        "LightGBM-Window": Tabular("LightGBM-Window", "std", "lgbm", seed, window=10),
        "LightGBM-Decay": Tabular("LightGBM-Decay", "std", "lgbm", seed, sample_weight="decay"),
        "LightGBM-DDM": DriftTriggered("LightGBM-DDM", "std", seed),
        "ARF": ARF("ARF", "std", seed),
    }
    if which == "core":
        keep = ["LR", "XGBoost", "LightGBM", "CatBoost", "RF", "LightGBM-AllViews", "LR-AllViews", "XGBoost-AllViews",
                "LightGBM-Decay"]
        B = {k: B[k] for k in keep}
    elif isinstance(which, (list, tuple)):
        B = {k: B[k] for k in which}
    return B


# Ensembles: name -> list of member learners; fused by the same history-trained
# logistic-regression planner (with context for AMIF-X, without for plain stacking).
ENSEMBLES = {
    "AMIF-X": {"members": ["FullProfile", "Transaction", "Neighbourhood", "Temporal", "Actor", "Graph", "Anomaly",
                           "NeighbourRisk"], "context": True},
    "Stacking": {"members": ["LR", "RF", "XGBoost", "LightGBM", "CatBoost"], "context": False},
    "Stacking-AllViews": {"members": ["LR-AllViews", "XGBoost-AllViews", "LightGBM-AllViews"], "context": False},
}
