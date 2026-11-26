"""Point-in-time case table for Elliptic++ (transaction-level cases).

Addresses reviewer point 4 (leakage). Rules enforced here:
  * No statistic is fitted on the whole data set. Every feature of a transaction at
    step t is a function of data observed at steps <= t only.
  * Graph features use only edges among transactions of the same time step. We verify
    below that every Elliptic++ edge connects two transactions of the same step, so the
    within-step graph is exactly the graph observable at the end of step t.
  * Wallet / actor evidence is computed from address<->transaction links cumulated up
    to and including step t. The Elliptic++ wallet feature table is NOT used because it
    contains whole-lifetime totals (future information). Address labels are never used.
  * Within-step normalisation (z-score, rank) uses the unlabelled feature values of the
    transactions of step t, which are observed when step t is investigated (batch
    setting: cases of step t are scored at the end of step t). No labels are used.
  * Standardisation for linear / neural models is fitted inside each learner on the
    training rows available at the refit time (see models/*.py), never here.

Output: cache/elliptic_cases.parquet, cache/elliptic_edges.parquet, cache/elliptic_meta.json
Column prefixes define evidence views:
  tx__  local transaction behaviour (93 local + 17 augmented Elliptic++ features)
  nb__  one-hop neighbourhood aggregates (72 Elliptic aggregate features)
  tp__  temporal context (within-step z-scores / ranks, step volume; no time index)
  ac__  causal wallet / actor evidence
  gs__  graph structure (within-step graph)
  ctx__ case context used by the planner (evidence availability, degree, volume)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from amifx.config import CACHE, RAW_ELLIPTIC


def _read(name, **kw):
    for nm in (name, "Copy of " + name):
        p = RAW_ELLIPTIC / nm
        if p.exists():
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(f"{name} not found in {RAW_ELLIPTIC}")


def build(force: bool = False, max_step: int | None = None):
    sfx = "" if max_step is None else f"_upto{max_step}"
    out = CACHE / f"elliptic_cases{sfx}.parquet"
    if out.exists() and not force:
        return out
    t0 = time.time()
    log = lambda *a: print(f"[elliptic {time.time() - t0:6.0f}s]", *a, flush=True)

    tx = _read("txs_features.csv", low_memory=False)
    cls = _read("txs_classes.csv")
    tx = tx.merge(cls, on="txId", how="left", validate="one_to_one")
    tx["y"] = np.where(tx["class"].eq(1), 1, np.where(tx["class"].eq(2), 0, -1)).astype("int8")
    tx = tx.rename(columns={"Time step": "step"}).drop(columns=["class"])
    if max_step is not None:                       # leakage test: delete all future data
        tx = tx[tx.step <= max_step]
    tx = tx.reset_index(drop=True)
    N = len(tx)
    log("transactions", N, "labelled", int((tx.y >= 0).sum()), "illicit", int((tx.y == 1).sum()))

    local = [c for c in tx.columns if c.startswith("Local_feature_")]
    aggr = [c for c in tx.columns if c.startswith("Aggregate_feature_")]
    extra = ["in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size", "num_input_addresses",
             "num_output_addresses", "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median",
             "in_BTC_total", "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total"]
    extra = [c for c in extra if c in tx.columns]
    feats = pd.DataFrame({"id": tx.txId.values, "step": tx.step.astype("int16").values, "y": tx.y.values})
    for c in local + extra:
        feats["tx__" + c] = tx[c].astype("float32").values
    for c in aggr:
        feats["nb__" + c] = tx[c].astype("float32").values

    # ---------------- graph (within-step) ----------------
    e = _read("txs_edgelist.csv")
    id2i = pd.Series(np.arange(N), index=tx.txId.values)
    s = id2i.reindex(e.txId1.values).values
    d = id2i.reindex(e.txId2.values).values
    ok = ~(np.isnan(s) | np.isnan(d))
    s, d = s[ok].astype(np.int64), d[ok].astype(np.int64)
    step = tx.step.values
    cross = int((step[s] != step[d]).sum())
    log("edges", len(s), "cross-step edges", cross)
    keep = step[s] == step[d]           # enforce within-step (no-op if cross == 0)
    s, d = s[keep], d[keep]
    indeg = np.bincount(d, minlength=N).astype("float32")
    outdeg = np.bincount(s, minlength=N).astype("float32")
    deg = indeg + outdeg
    feats["gs__in_deg"] = indeg
    feats["gs__out_deg"] = outdeg
    feats["gs__deg"] = deg
    feats["gs__fanin_ratio"] = indeg / (deg + 1)

    def grp(idx, val, fn):
        sr = pd.Series(val).groupby(idx)
        r = getattr(sr, fn)()
        o = np.zeros(N, "float32")
        o[r.index.values] = r.values
        return o

    feats["gs__pred_mean_out"] = grp(d, outdeg[s], "mean")
    feats["gs__pred_max_out"] = grp(d, outdeg[s], "max")
    feats["gs__succ_mean_in"] = grp(s, indeg[d], "mean")
    feats["gs__succ_max_in"] = grp(s, indeg[d], "max")
    u = np.concatenate([s, d]); v = np.concatenate([d, s])
    feats["gs__2hop"] = grp(u, deg[v], "sum")
    feats["gs__nb_mean_deg"] = grp(u, deg[v], "mean")
    # connected-component size within the step graph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(len(s)), (s, d)), shape=(N, N))
    _, comp = connected_components(A, directed=False)
    feats["gs__cc_size"] = pd.Series(comp).map(pd.Series(comp).value_counts()).values.astype("float32")

    # ---------------- temporal context (within-step, label-free) ----------------
    feats["tp__step_volume"] = feats.groupby("step").id.transform("count").astype("float32")
    for c in ["total_BTC", "fees", "size", "num_input_addresses", "num_output_addresses", "in_BTC_total",
              "out_BTC_total", "in_txs_degree", "out_txs_degree"]:
        if c not in tx.columns:
            continue
        g = tx.groupby("step")[c]
        feats["tp__z_" + c] = ((tx[c] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype("float32").values
        feats["tp__rank_" + c] = g.rank(pct=True).astype("float32").values
    # relative fee rate and value concentration vs. the step
    fr = (tx["fees"] / (tx["size"] + 1)).astype("float32")
    feats["tp__rank_feerate"] = fr.groupby(tx.step).rank(pct=True).astype("float32").values
    log("graph + temporal views done")

    # ---------------- causal wallet / actor evidence ----------------
    tt = pd.DataFrame({"txId": tx.txId.values, "step": tx.step.values, "btc": tx["total_BTC"].values})
    ins = _read("AddrTx_edgelist.csv").rename(columns={"input_address": "addr"}).merge(tt, on="txId")
    outs = _read("TxAddr_edgelist.csv").rename(columns={"output_address": "addr"}).merge(tt, on="txId")
    snd = ins.groupby(["addr", "step"]).agg(s_n=("txId", "size"), s_btc=("btc", "sum")).reset_index()
    rcv = outs.groupby(["addr", "step"]).agg(r_n=("txId", "size"), r_btc=("btc", "sum")).reset_index()
    act = snd.merge(rcv, on=["addr", "step"], how="outer").fillna(0).sort_values(["addr", "step"])
    g = act.groupby("addr", sort=False)
    for c in ["s_n", "s_btc", "r_n", "r_btc"]:
        act["cum_" + c] = g[c].cumsum()
    act["active_steps"] = g.cumcount() + 1
    act["age"] = act["step"] - g["step"].transform("min")          # min over steps <= current row
    act["new_addr"] = (act.age == 0).astype("float32")
    act["recv_send_ratio"] = (act.cum_r_n + 1) / (act.cum_s_n + 1)
    act["step_n"] = act.s_n + act.r_n
    act["reuse"] = (act.cum_s_n + act.cum_r_n > 1).astype("float32")
    keepc = ["cum_s_n", "cum_s_btc", "cum_r_n", "cum_r_btc", "active_steps", "age", "new_addr",
             "recv_send_ratio", "step_n", "reuse"]
    act[keepc] = act[keepc].astype("float32")
    parts = []
    for ed, pref in [(ins, "ac__in_"), (outs, "ac__out_")]:
        m = ed[["addr", "txId", "step"]].merge(act[["addr", "step"] + keepc], on=["addr", "step"], how="left")
        a = m.groupby("txId")[keepc].agg(["mean", "max"])
        a.columns = [pref + "_".join(c) for c in a.columns]
        a[pref + "n_addr"] = m.groupby("txId").size().astype("float32")
        a[pref + "share_new"] = m.groupby("txId")["new_addr"].mean().astype("float32")
        parts.append(a)
    cw = pd.concat(parts, axis=1)
    cw = cw.reindex(tx.txId.values)
    for c in cw.columns:
        feats[c] = cw[c].values.astype("float32")
    # wallet evidence availability (some transactions have no address links in Elliptic++)
    nin = np.nan_to_num(feats["ac__in_n_addr"].values)
    nout = np.nan_to_num(feats["ac__out_n_addr"].values)
    log("actor view done; tx with address links:", float(((nin + nout) > 0).mean()))

    # ---------------- planner context ----------------
    feats["ctx__log_deg"] = np.log1p(deg)
    feats["ctx__has_wallet"] = ((nin + nout) > 0).astype("float32")
    feats["ctx__log_naddr"] = np.log1p(nin + nout).astype("float32")
    feats["ctx__log_volume"] = np.log1p(feats["tp__step_volume"].values)

    feats.to_parquet(out)
    pd.DataFrame({"src": s, "dst": d}).to_parquet(CACHE / f"elliptic_edges{sfx}.parquet")
    views = {p: [c for c in feats.columns if c.startswith(p)] for p in ["tx__", "nb__", "tp__", "ac__", "gs__", "ctx__"]}
    meta = {"n_nodes": int(N), "n_edges": int(len(s)), "cross_step_edges": cross,
            "n_labelled": int((feats.y >= 0).sum()), "n_illicit": int((feats.y == 1).sum()),
            "n_licit": int((feats.y == 0).sum()), "views": {k: len(v) for k, v in views.items()},
            "seconds": round(time.time() - t0, 1)}
    (CACHE / f"elliptic_meta{sfx}.json").write_text(json.dumps(meta, indent=1))
    log("saved", feats.shape, meta)
    return out


if __name__ == "__main__":
    import sys
    build(force="--force" in sys.argv)
