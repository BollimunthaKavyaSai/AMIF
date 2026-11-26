"""Point-in-time case table for IBM AMLSim (account-window cases).

A case is an (account, 10-day window) pair in which the account was active. The case
is suspicious (y=1) if the account sent or received at least one transaction that the
simulator generated as part of a money-laundering typology (fan-in or cycle) in that
window. This mirrors how AML alerts are raised on accounts over review periods.

Point-in-time rules (same as Elliptic++):
  * Window features use only transactions of window w.
  * Actor / history features use only windows < w (strictly past).
  * The graph of window w contains only transactions of window w.
  * accounts.csv IS_FRAUD and TX_BEHAVIOR_ID are never used (label / simulator internals).
    ALERT_ID / IS_FRAUD of transactions are used only to build the label.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from amifx.config import CACHE, RAW_AMLSIM

WINDOW_DAYS = 10


def build(name: str = "amlsim10k", force: bool = False, max_step: int | None = None):
    sfx = "" if max_step is None else f"_upto{max_step}"
    out = CACHE / f"{name}_cases{sfx}.parquet"
    if out.exists() and not force:
        return out
    t0 = time.time()
    log = lambda *a: print(f"[{name} {time.time() - t0:6.0f}s]", *a, flush=True)
    raw = RAW_AMLSIM[name]
    t = pd.read_csv(raw / "transactions.csv",
                    usecols=["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TIMESTAMP", "IS_FRAUD"],
                    dtype={"SENDER_ACCOUNT_ID": "int32", "RECEIVER_ACCOUNT_ID": "int32", "TX_AMOUNT": "float64",
                           "TIMESTAMP": "int16"})
    t["IS_FRAUD"] = t["IS_FRAUD"].astype(str).str.lower().eq("true").astype("int8")
    t.columns = ["s", "r", "amt", "day", "fraud"]
    t["w"] = (t.day // WINDOW_DAYS + 1).astype("int16")
    t["la"] = np.log1p(t.amt).astype("float32")
    if max_step is not None:                       # leakage test: delete all future data
        t = t[t.w <= max_step].reset_index(drop=True)
    acc = pd.read_csv(raw / "accounts.csv", usecols=["ACCOUNT_ID", "INIT_BALANCE"])
    bal = acc.set_index("ACCOUNT_ID").INIT_BALANCE
    log("transactions", len(t), "fraud tx", int(t.fraud.sum()))

    # ---- per account-window activity (sent / received) ----
    def side(col, other, pref):
        g = t.groupby([col, "w"])
        a = g.agg(n=("amt", "size"), sum=("amt", "sum"), lmean=("la", "mean"), lmax=("la", "max"),
                  lmin=("la", "min"), lstd=("la", "std"), days=("day", "nunique"), ncp=(other, "nunique"))
        burst = t.groupby([col, "w", "day"]).size().groupby(level=[0, 1]).max()
        a["burst"] = burst
        a["lsum"] = np.log1p(a["sum"])
        a = a.drop(columns="sum")
        a.columns = [pref + c for c in a.columns]
        a.index.names = ["acct", "w"]
        return a

    snd = side("s", "r", "s_")
    rcv = side("r", "s", "r_")
    cw = snd.join(rcv, how="outer")
    cw = cw.fillna({c: 0 for c in cw.columns if not c.endswith(("lstd", "lmean", "lmax", "lmin"))})
    cw = cw.reset_index()
    # label: any fraud transaction involving the account in the window
    fs = t[t.fraud == 1]
    lab = pd.concat([fs[["s", "w"]].rename(columns={"s": "acct"}), fs[["r", "w"]].rename(columns={"r": "acct"})]).drop_duplicates()
    lab["y"] = 1
    cw = cw.merge(lab, on=["acct", "w"], how="left")
    cw["y"] = cw.y.fillna(0).astype("int8")
    cw = cw.sort_values(["w", "acct"]).reset_index(drop=True)
    N = len(cw)
    log("account-window cases", N, "suspicious", int(cw.y.sum()))
    F = {"id": (cw.acct.astype("int64") * 100 + cw.w).values, "step": cw.w.astype("int16").values, "y": cw.y.values}

    # ---- transaction view (own window activity) ----
    own = [c for c in cw.columns if c.startswith(("s_", "r_"))]
    for c in own:
        F["tx__" + c] = cw[c].astype("float32").values
    F["tx__net_flow"] = (np.expm1(cw.r_lsum) - np.expm1(cw.s_lsum)).astype("float32").values
    F["tx__in_out_ratio"] = ((cw.r_n + 1) / (cw.s_n + 1)).astype("float32").values
    F["tx__log_init_balance"] = np.log1p(cw.acct.map(bal).clip(lower=0)).astype("float32").values

    # ---- temporal view: within-window normalisation + change vs. previous window ----
    for c in ["s_n", "r_n", "s_lsum", "r_lsum", "s_lmean", "r_lmean", "s_ncp", "r_ncp", "s_burst", "r_burst"]:
        g = cw.groupby("w")[c]
        F["tp__z_" + c] = ((cw[c] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype("float32").values
        F["tp__rank_" + c] = g.rank(pct=True).astype("float32").values
    prev = cw[["acct", "w", "s_n", "r_n", "s_lsum", "r_lsum"]].copy()
    prev["w"] = prev.w + 1
    pm = cw[["acct", "w"]].merge(prev, on=["acct", "w"], how="left")
    for c in ["s_n", "r_n", "s_lsum", "r_lsum"]:
        F["tp__delta_" + c] = (cw[c].values - pm[c].fillna(0).values).astype("float32")
    F["tp__prev_active"] = pm.s_n.notna().astype("float32").values
    F["tp__window_volume"] = cw.groupby("w").acct.transform("count").astype("float32").values

    # ---- actor / history view: strictly past windows ----
    hist = cw[["acct", "w", "s_n", "r_n", "s_ncp", "r_ncp"]].copy()
    hist["s_amt"] = np.expm1(cw.s_lsum.values)
    hist["r_amt"] = np.expm1(cw.r_lsum.values)
    hist = hist.sort_values(["acct", "w"])
    g = hist.groupby("acct", sort=False)
    for c in ["s_n", "r_n", "s_amt", "r_amt", "s_ncp", "r_ncp"]:
        hist["cum_" + c] = g[c].cumsum() - hist[c]             # exclusive of current window
    hist["n_prev_windows"] = g.cumcount()
    hist["first_w"] = g.w.transform("min")
    hist["age"] = hist.w - hist.first_w
    hist = hist.set_index(["acct", "w"]).reindex(pd.MultiIndex.from_arrays([cw.acct, cw.w]))
    for c in ["cum_s_n", "cum_r_n", "cum_s_amt", "cum_r_amt", "cum_s_ncp", "cum_r_ncp"]:
        F["ac__log_" + c] = np.log1p(hist[c].values).astype("float32")
    F["ac__n_prev_windows"] = hist.n_prev_windows.values.astype("float32")
    F["ac__age"] = hist.age.values.astype("float32")
    F["ac__new_account"] = (hist.n_prev_windows.values == 0).astype("float32")
    F["ac__hist_in_out"] = ((hist.cum_r_n.values + 1) / (hist.cum_s_n.values + 1)).astype("float32")
    pw = np.maximum(hist.n_prev_windows.values, 1)
    F["ac__rate_change_s"] = (cw.s_n.values / ((hist.cum_s_n.values / pw) + 1)).astype("float32")
    F["ac__rate_change_r"] = (cw.r_n.values / ((hist.cum_r_n.values / pw) + 1)).astype("float32")
    log("tx / temporal / actor views done")

    # ---- graph of each window (account-window nodes, unique directed pairs) ----
    key = pd.Series(np.arange(N), index=pd.MultiIndex.from_arrays([cw.acct.values, cw.w.values]))
    pairs = t.groupby(["s", "r", "w"]).size().reset_index(name="cnt")
    si = key.reindex(pd.MultiIndex.from_arrays([pairs.s.values, pairs.w.values])).values
    di = key.reindex(pd.MultiIndex.from_arrays([pairs.r.values, pairs.w.values])).values
    s_, d_ = si.astype(np.int64), di.astype(np.int64)
    indeg = np.bincount(d_, minlength=N).astype("float32")
    outdeg = np.bincount(s_, minlength=N).astype("float32")
    deg = indeg + outdeg
    F["gs__in_deg"], F["gs__out_deg"], F["gs__deg"] = indeg, outdeg, deg
    F["gs__fanin_ratio"] = indeg / (deg + 1)
    # reciprocity: counterparties in both directions
    pr = pd.DataFrame({"a": s_, "b": d_})
    rec = pr.merge(pr.rename(columns={"a": "b", "b": "a"}), on=["a", "b"])
    F["gs__reciprocal"] = np.bincount(rec.a.values, minlength=N).astype("float32")

    def grp(idx, val, fn):
        r = getattr(pd.Series(val).groupby(idx), fn)()
        o = np.zeros(N, "float32"); o[r.index.values] = r.values
        return o

    u = np.concatenate([s_, d_]); v = np.concatenate([d_, s_])
    F["gs__2hop"] = grp(u, deg[v], "sum")
    F["gs__nb_mean_deg"] = grp(u, deg[v], "mean")
    F["gs__pred_mean_out"] = grp(d_, outdeg[s_], "mean")
    F["gs__succ_mean_in"] = grp(s_, indeg[d_], "mean")
    # strongly connected component size (cycles) within the window graph
    A = coo_matrix((np.ones(len(s_)), (s_, d_)), shape=(N, N)).tocsr()
    _, comp = connected_components(A, directed=True, connection="strong")
    cs = pd.Series(comp).map(pd.Series(comp).value_counts()).values.astype("float32")
    F["gs__log_scc_size"] = np.log1p(cs)
    # two-step return paths (a->b->a is reciprocity; a->b->c->a counted via A^3 diagonal, sparse)
    A1 = (A > 0).astype("float32")
    A2 = A1 @ A1
    F["gs__cycle3"] = np.asarray((A2.multiply(A1.T)).sum(axis=1)).ravel().astype("float32")

    # ---- neighbourhood view: aggregates of counterparties' own window activity ----
    base = np.vstack([cw[c].fillna(0).values for c in ["s_n", "r_n", "s_lmean", "r_lmean", "s_ncp", "r_ncp"]]).T
    names = ["s_n", "r_n", "s_lmean", "r_lmean", "s_ncp", "r_ncp"]
    for j, nm in enumerate(names):
        F["nb__mean_" + nm] = grp(u, base[v, j], "mean")
        F["nb__max_" + nm] = grp(u, base[v, j], "max")
    F["nb__mean_deg"] = grp(u, deg[v], "mean")
    F["nb__share_new"] = grp(u, F["ac__new_account"][v], "mean")
    log("graph + neighbourhood views done")

    # ---- planner context ----
    F["ctx__log_deg"] = np.log1p(deg)
    F["ctx__has_history"] = (1 - F["ac__new_account"]).astype("float32")
    F["ctx__log_ncp"] = np.log1p(cw.s_ncp.values + cw.r_ncp.values).astype("float32")
    F["ctx__log_volume"] = np.log1p(F["tp__window_volume"])

    feats = pd.DataFrame(F)
    feats.to_parquet(out)
    pd.DataFrame({"src": s_, "dst": d_}).to_parquet(CACHE / f"{name}_edges{sfx}.parquet")
    views = {p: [c for c in feats.columns if c.startswith(p)] for p in ["tx__", "nb__", "tp__", "ac__", "gs__", "ctx__"]}
    per = feats.groupby("step").y.agg(["size", "sum"]).reset_index()
    meta = {"n_cases": int(N), "n_suspicious": int(feats.y.sum()), "n_edges": int(len(s_)),
            "n_transactions": int(len(t)), "n_fraud_tx": int(t.fraud.sum()),
            "views": {k: len(v) for k, v in views.items()}, "per_window": per.values.tolist(),
            "seconds": round(time.time() - t0, 1)}
    (CACHE / f"{name}_meta{sfx}.json").write_text(json.dumps(meta, indent=1))
    log("saved", feats.shape, {k: meta[k] for k in ["n_cases", "n_suspicious", "n_edges", "views"]})
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "amlsim10k", force="--force" in sys.argv)
