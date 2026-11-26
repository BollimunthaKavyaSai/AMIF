"""Prequential (test-then-train) protocol with simulated analyst feedback.

For every stream step t:
  1. every learner is (re)fitted on the cases whose *observed* labels are available at t;
  2. it scores the cases of step t (out-of-sample);
  3. ensembles fuse member scores with a planner fitted on out-of-sample history;
  4. the decision threshold of EVERY model (baselines included) is the F1-optimal
     threshold on that model's own out-of-sample history with observed labels;
  5. analysts review cases of step t according to the feedback scenario, and the
     reviewed labels (possibly noisy) become available at step t + 1 + delay.

Static protocol: models are fitted once at the end of the initial period and never
updated; their threshold is chosen on out-of-fold scores of the initial period (no
fixed 0.5 threshold anywhere).

The initial period (steps <= initial_end) is historical labelled data. Out-of-fold
scores on it (3 folds grouped by time step) seed the history used by planners and
thresholds, identically for all models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amifx.config import FEEDBACK, PRED_DIR, SPLITS
from amifx.data.loader import load
from amifx.models.learners import ENSEMBLES, GraphAgent, Tabular, make_agents, make_baselines
from amifx.planner import Planner, best_f1_threshold

INF = 10 ** 6


def _folds(steps, k=3):
    u = np.unique(steps)
    return {s: i % k for i, s in enumerate(u)}


class Run:
    def __init__(self, dataset, split="main", scenario="oracle", seed=0, baselines="all", static=False,
                 agents=True, ensembles=None, tag=None, log=True, cvcl=False, review="committee", suffix="", sym_filter=False, amx=None):
        self.D = load(dataset)
        self.S = SPLITS[dataset][split]
        self.fb = FEEDBACK[scenario]
        self.seed, self.static = seed, static
        self.cvcl, self.review, self.sym_filter = cvcl, review, sym_filter
        # AMIF-X method options (CVCL v2); defaults reproduce CVCL v1
        self.amx = dict(loo=False, pseudo_w=1.0, decay=False, full_view="std", planner_decay=False, explore=0.0, clean_hist=False, pseudo="both", planner="learned", nr_view="all", graph_model="lgbm", collective_agent=False)
        self.amx.update(amx or {})
        self.dataset, self.split, self.scenario = dataset, split, ("static" if static else scenario)
        suffix = ("" if review == "committee" else f"-{review}") + ("-symfilter" if sym_filter else "") + suffix
        self.tag = tag or f"{dataset}__{split}__{self.scenario}{suffix}__s{seed}"
        self.verbose = log
        self.t0 = time.time()
        comps = {}
        if agents:
            comps.update(make_agents(seed, self.amx["full_view"], self.amx["decay"], self.amx["nr_view"], self.amx["graph_model"], self.amx["collective_agent"]))
        comps.update(make_baselines(seed, baselines))
        self.protos = comps
        ens = ensembles if ensembles is not None else [e for e, c in ENSEMBLES.items() if all(m in comps for m in c["members"])]
        self.ensembles = {e: dict(ENSEMBLES[e]) for e in ens}
        if "AMIF-X" in self.ensembles and "CollectiveRisk" in comps:
            self.ensembles["AMIF-X"]["members"] = self.ensembles["AMIF-X"]["members"] + ["CollectiveRisk"]
        self.shared = self.fb.budget is None          # feedback independent of scores -> one shared label mask
        # groups: name -> members
        self.groups = {}
        for e, cfg in self.ensembles.items():
            self.groups[e] = cfg["members"]
        for c in comps:
            if self.shared or c not in make_agents(0):
                self.groups.setdefault(c, [c])
        if not self.shared:     # under budgeted review, agents only matter inside AMIF-X; members of stacks are per-group
            self.groups = {g: m for g, m in self.groups.items() if g in self.ensembles or g in make_baselines(0, baselines)}

    def log(self, *a):
        if self.verbose:
            print(f"[{self.tag} {time.time() - self.t0:7.0f}s]", *a, flush=True)

    # ------------------------------------------------------------------ helpers
    def _key(self, g, c):
        return c if self.shared else f"{g}/{c}"

    def run(self):
        D, S, fb = self.D, self.S, self.fb
        N, step, y = D.n, D.step, D.y.astype(int)
        cases = np.where(y >= 0)[0]
        rng = np.random.default_rng(10_000 + self.seed)
        u_noise, u_audit = rng.random(N), rng.random(N)
        flip = ((y == 1) & (u_noise < fb.fn_rate)) | ((y == 0) & (u_noise < fb.fp_rate))
        y_noisy = np.where(flip, 1 - y, y)
        init = cases[step[cases] <= S.initial_end]
        masks = ["shared"] if self.shared else list(self.groups)
        avail = {m: np.full(N, INF) for m in masks}
        yobs = {m: np.full(N, -1) for m in masks}
        for m in masks:
            avail[m][init] = 0; yobs[m][init] = y[init]
        reviewed = {m: np.zeros(N, bool) for m in masks}
        mask_of = (lambda g: "shared") if self.shared else (lambda g: g)

        # component instances
        keys = {}
        for g, mem in self.groups.items():
            for c in mem:
                keys[self._key(g, c)] = (c, mask_of(g))
        inst = {k: self.protos[c].clone() for k, (c, _) in keys.items()}
        cscore = {k: np.full(N, np.nan, "float32") for k in keys}
        gscore = {g: np.full(N, np.nan, "float32") for g in self.groups}
        timing = {k: {"fit": [], "predict": [], "n_pred": []} for k in keys}
        thresholds = {g: {} for g in self.groups}
        coefs = {g: {} for g in self.ensembles}
        last_sig = {k: None for k in keys}

        # ---------------- out-of-fold seeding on the initial period ----------------
        fo = _folds(step[init]); fold = np.array([fo[s] for s in step[init]])
        seeded = {}
        for k, (c, m) in keys.items():
            proto_key = c if self.shared else c   # identical labels in the initial period -> share across groups
            if proto_key in seeded:
                cscore[k][init] = seeded[proto_key]; continue
            t1 = time.time()
            if self.protos[c].streaming:                      # test-then-train over the initial steps
                L = self.protos[c].clone(); out = np.full(len(init), np.nan, "float32")
                for s in np.unique(step[init]):
                    sel = step[init] == s
                    out[sel] = L.predict(D, init[sel])
                    L.fit(D, init[step[init] <= s], y[init[step[init] <= s]], s + 1)
            else:
                out = np.full(len(init), np.nan, "float32")
                for f in range(3):
                    L = self.protos[c].clone()
                    L.fit(D, init[fold != f], y[init[fold != f]], S.initial_end + 1)
                    out[fold == f] = L.predict(D, init[fold == f])
            seeded[proto_key] = out
            cscore[k][init] = out
            self.log(f"OOF seed {c:18s} {time.time() - t1:6.1f}s")
        for g, mem in self.groups.items():
            if g in self.ensembles:
                P = np.column_stack([cscore[self._key(g, c)][init] for c in mem])
                Z = D.ctx()[init] if self.ensembles[g]["context"] else None
                out = np.full(len(init), np.nan)
                for f in range(3):
                    pl = Planner(context=Z is not None).fit(P[fold != f], y[init[fold != f]], None if Z is None else Z[fold != f])
                    out[fold == f] = pl.score(P[fold == f], None if Z is None else Z[fold == f])
                gscore[g][init] = out
            else:
                gscore[g][init] = cscore[self._key(g, mem[0])][init]

        # ---------------- Cross-View Consensus Learning (CVCL) set-up ----------------
        # owner of the label-augmentation rule for each component key
        owner = {}
        for k, (c, m) in keys.items():
            g = k.split("/")[0] if not self.shared else None
            if self.cvcl and "AMIF-X" in self.groups and (g == "AMIF-X" or (self.shared and c in self.groups["AMIF-X"])):
                owner[k] = "AMIF-X"
            elif (g or c) == "LightGBM-SelfTrain":
                owner[k] = "LightGBM-SelfTrain"
        tau = {}
        base_rate = y[init].mean()
        for o in set(owner.values()):
            s = gscore[o][init]; o_ = np.argsort(-s); prec = np.cumsum(y[init][o_]) / np.arange(1, len(o_) + 1)
            ok = np.where((prec >= 0.97) & (np.arange(1, len(o_) + 1) >= 20))[0]
            hi = float(s[o_][ok.max()]) if len(ok) else 1.0
            o2 = np.argsort(s); rate = np.cumsum(y[init][o2]) / np.arange(1, len(o2) + 1)
            ok2 = np.where(rate <= 0.02 * base_rate)[0]           # pseudo-negatives: <= 2% of the base error rate
            lo = float(s[o2][ok2.max()]) if len(ok2) else 0.0
            tau[o] = (hi, min(lo, 0.5 * hi))
        self.log("CVCL thresholds", tau)
        cvcl_stats = {o: {} for o in tau}

        def consensus(o, rows):
            r = gscore[o][rows]
            hi, lo = tau[o]
            if o == "AMIF-X":
                P = np.column_stack([cscore[self._key("AMIF-X", c)][rows] for c in self.groups["AMIF-X"]])
                pos = (r >= hi) & ((P >= 0.5).sum(1) >= int(np.ceil(0.75 * P.shape[1])))
                neg = (r <= lo) & (P <= 0.5).all(1)
            else:
                pos, neg = r >= hi, r <= lo
            return pos & ~np.isnan(r), neg & ~np.isnan(r)

        last_pl = {}

        def loo_consensus(c, rows):
            """Leave-one-view-out consensus for agent c: the planner log-odds without agent c's exact
            Shapley contribution (co-training: agent c is taught by the other views)."""
            pl = last_pl.get("AMIF-X")
            mem = self.groups["AMIF-X"]
            if pl is None or getattr(pl, "kind_fallback", False):
                return consensus("AMIF-X", rows)
            P = np.column_stack([cscore[self._key("AMIF-X", x)][rows] for x in mem])
            Z = D.ctx()[rows]
            ok = ~np.isnan(P).any(1)
            P = np.where(np.isnan(P), 0.5, P)
            L = np.log(np.clip(pl.score(P, Z), 1e-6, 1 - 1e-6)); L = L - np.log(np.clip(1 - np.exp(L), 1e-6, 1))
            phi = pl.contributions(P, Z)[0][:, mem.index(c)]
            r = 1 / (1 + np.exp(-(L - phi)))
            hi, lo = tau["AMIF-X"]
            return (r >= hi) & ok, (r <= lo) & ok

        def training_set(k, m, t):
            tr = np.where(avail[m] <= t)[0]
            o = owner.get(k)
            if o is None or self.static:
                return tr, yobs[m][tr], None
            c = keys[k][0]
            ck = c if (o == "AMIF-X" and self.amx["loo"]) else "_all"
            if (t, ck) in cvcl_stats[o].get("_cache", {}):
                return cvcl_stats[o]["_cache"][(t, ck)]
            st_rows = cases[(step[cases] > S.initial_end) & (step[cases] < t)]
            if ck != "_all":
                pos, neg = loo_consensus(c, st_rows)
            else:
                pos, neg = consensus(o, st_rows)
            av = avail[o if not self.shared else m][st_rows] <= t
            yo = yobs[o if not self.shared else m][st_rows]
            # consensus noise filter (asymmetric): only recorded-licit labels contradicted by a positive
            # consensus are dropped. Recorded-illicit labels are never dropped: missed laundering is the
            # dominant analyst error, and illicit cases that all views score low are typically *new*
            # fraud patterns (concept drift) that the agents must learn, not noise.
            if self.sym_filter:
                drop = av & (((yo == 0) & pos) | ((yo == 1) & neg))
            else:
                drop = av & (yo == 0) & pos
            if o == "AMIF-X" and self.amx["pseudo"] == "pos":
                neg = np.zeros_like(neg)                                   # positive-only pseudo-labels
            elif o == "AMIF-X" and self.amx["pseudo"] == "none":
                pos, neg = pos & False, neg & False
            pseudo = ~av & (pos | neg)                                     # cross-view pseudo-labels
            keep = np.setdiff1d(tr, st_rows[drop])
            rows = np.concatenate([keep, st_rows[pseudo]])
            lab = np.concatenate([yobs[m][keep], pos[pseudo].astype(int)])
            pw = self.amx["pseudo_w"] if o == "AMIF-X" else 1.0
            w = None if pw == 1.0 else np.concatenate([np.ones(len(keep)), np.full(int(pseudo.sum()), pw)])
            cvcl_stats[o].setdefault("_cache", {})[(t, ck)] = (rows, lab, w)
            if ck not in ("_all", "FullProfile"):
                return rows, lab, w
            cvcl_stats[o][t] = {"dropped": int(drop.sum()),
                                "dropped_correct": int((drop & (yo == y[st_rows])).sum()),
                                "pseudo": int(pseudo.sum()), "pseudo_pos": int((pseudo & pos).sum()),
                                "pseudo_wrong": int((pseudo & (pos.astype(int) != y[st_rows])).sum())}
            return rows, lab, w

        # ---------------- stream ----------------
        for t in S.stream:
            cur = cases[step[cases] == t]
            for k, (c, m) in keys.items():
                tr, ytr, wtr = training_set(k, m, t)
                sig = (len(tr), int(tr.sum()) if len(tr) else 0, int(ytr.sum()))
                L = inst[k]
                t1 = time.time()
                if (not self.static or last_sig[k] is None) and sig != last_sig[k]:
                    if wtr is not None and isinstance(L, (Tabular, GraphAgent)):
                        L.fit(D, tr, ytr, t, w=wtr)
                    else:
                        L.fit(D, tr, ytr, t)
                    last_sig[k] = sig
                    timing[k]["fit"].append(time.time() - t1)
                t2 = time.time()
                cscore[k][cur] = L.predict(D, cur)
                timing[k]["predict"].append(time.time() - t2); timing[k]["n_pred"].append(len(cur))
            for g, mem in self.groups.items():
                m = mask_of(g)
                hist = np.where((avail[m] <= t) & (yobs[m] >= 0))[0]
                if g == "AMIF-X" and self.amx["clean_hist"] and self.cvcl and not self.static:
                    st_rows = cases[(step[cases] > S.initial_end) & (step[cases] < t)]
                    pos_, _ = consensus("AMIF-X", st_rows)
                    bad = st_rows[pos_ & (avail[m][st_rows] <= t) & (yobs[m][st_rows] == 0)]
                    hist = np.setdiff1d(hist, bad)
                if g in self.ensembles:
                    ctx = self.ensembles[g]["context"]
                    P = np.column_stack([cscore[self._key(g, c)] for c in mem])
                    ok = hist[~np.isnan(P[hist]).any(1)]
                    pw_ = None
                    if g == "AMIF-X" and self.amx["planner_decay"]:
                        pw_ = 0.5 ** ((t - 1 - step[ok]).clip(min=0) / 5.0)
                    kind_ = self.amx["planner"] if g == "AMIF-X" else "learned"
                    pl = Planner(kind=kind_, context=ctx).fit(P[ok], yobs[m][ok], D.ctx()[ok] if ctx else None, w=pw_)
                    gscore[g][cur] = pl.score(P[cur], D.ctx()[cur] if ctx else None)
                    last_pl[g] = pl
                    if not getattr(pl, "kind_fallback", False):
                        b, gm = pl.weights()
                        coefs[g][int(t)] = {"beta": dict(zip(mem, map(float, b))),
                                            "gamma": dict(zip(D.views["ctx"], map(float, gm))),
                                            "intercept": float(pl.lr.intercept_[0])}
                    # stacked models: threshold on the CURRENT planner's scores of the out-of-sample member
                    # history (consistent calibration; the planner is a low-capacity layer over OOS scores)
                    thresholds[g][int(t)] = best_f1_threshold(yobs[m][ok], pl.score(P[ok], D.ctx()[ok] if ctx else None))
                    continue
                else:
                    gscore[g][cur] = cscore[self._key(g, mem[0])][cur]
                h = hist[~np.isnan(gscore[g][hist])]
                thresholds[g][int(t)] = best_f1_threshold(yobs[m][h], gscore[g][h])
            # ---- analyst feedback on step t ----
            if not self.static:
                for m in masks:
                    if fb.budget is None:
                        rev = cur
                    else:
                        s = gscore[m][cur]
                        k_ = int(np.ceil(fb.budget * len(cur)))
                        explore = m in ("AMIF-X", "LightGBM-SelfTrain") and self.review == "committee"
                        k_top = int(np.ceil((1 - self.amx["explore"]) * k_)) if explore else k_
                        order = np.argsort(-s, kind="stable")
                        top = cur[order[:k_top]]
                        if explore and k_ > k_top:
                            rest = cur[order[k_top:]]
                            if m == "AMIF-X":       # committee disagreement across evidence views
                                P = np.column_stack([cscore[self._key("AMIF-X", c)][rest] for c in self.groups["AMIF-X"]])
                                u = np.std(np.log(np.clip(P, 1e-5, 1 - 1e-5) / (1 - np.clip(P, 1e-5, 1 - 1e-5))), 1)
                            else:                   # single-model uncertainty sampling (closest to threshold)
                                u = -np.abs(np.log(np.clip(gscore[m][rest], 1e-5, 1)) - np.log(max(thresholds[m][int(t)], 1e-5)))
                            top = np.union1d(top, rest[np.argsort(-u, kind="stable")[:k_ - k_top]])
                        rev = np.union1d(top, cur[u_audit[cur] < fb.audit])
                    reviewed[m][rev] = True
                    avail[m][rev] = t + 1 + fb.delay
                    yobs[m][rev] = y_noisy[rev]
            self.log(f"step {t} done")

        # ---------------- save ----------------
        keep = cases[step[cases] > S.initial_end - 1000]          # all labelled cases (initial OOF + stream)
        df = pd.DataFrame({"row": keep, "step": step[keep], "y": y[keep], "y_noisy": y_noisy[keep]})
        cols = {}
        for k in keys:
            cols["c:" + k] = cscore[k][keep]
        for g in self.groups:
            cols["g:" + g] = gscore[g][keep]
        if not self.shared:
            for m in masks:
                cols["r:" + m] = reviewed[m][keep]
        else:
            cols["r:shared"] = reviewed["shared"][keep]
        df = pd.concat([df, pd.DataFrame(cols)], axis=1)
        out = PRED_DIR / f"{self.tag}.parquet"
        df.to_parquet(out)
        info = {"dataset": self.dataset, "split": self.split, "scenario": self.scenario, "seed": self.seed,
                "feedback": fb.__dict__, "static": self.static, "groups": self.groups,
                "ensembles": {e: c for e, c in self.ensembles.items()}, "thresholds": thresholds, "coefs": coefs,
                "timing": {k: {a: (list(map(float, v)) if a != "n_pred" else v) for a, v in d.items()} for k, d in timing.items()},
                "refits": {k: getattr(inst[k], "n_retrain", None) for k in keys},
                "cvcl": {o: {int(t): v for t, v in d.items() if t != "_cache"} for o, d in cvcl_stats.items()}, "amx": self.amx,
                "cvcl_tau": tau, "cvcl_on": self.cvcl, "review": self.review,
                "ctx_names": self.D.views["ctx"], "seconds": time.time() - self.t0,
                "split_def": {"initial_end": S.initial_end, "test_start": S.test_start, "last": S.last}}
        (PRED_DIR / f"{self.tag}.json").write_text(json.dumps(info, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
        self.log("saved", out.name)
        return out
