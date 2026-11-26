"""Turn saved runs into every table / figure / number of the paper.

  python scripts/analyse.py            # everything available
Outputs: results/tables/*.csv, results/figures/*.png, results/numbers.json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from amifx import stats as st  # noqa: E402
from amifx.config import AGENTS, FIGURES, RESULTS, SHUTDOWN_STEP, TABLES, TIER1  # noqa: E402
from amifx.evaluate import (MAIN_METRICS, available, group_arrays, load_run, per_seed_metrics, pooled_boot,  # noqa: E402
                            replay_arrays, summary_table, tag)

B = int(__import__("os").environ.get("AMIFX_BOOT", 1000))
NUM = {}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

ORDER = ["AMIF-X", "LightGBM-Collective", "Stacking", "Stacking-AllViews", "LightGBM-AllViews", "XGBoost-AllViews", "LR-AllViews",
         "LightGBM", "XGBoost", "CatBoost", "RF", "LR", "MLP", "GCN", "GraphSAGE", "GAT", "SGC",
         "LightGBM-Window", "LightGBM-Decay", "LightGBM-DDM", "ARF", "LightGBM-SelfTrain", "LightGBM-Collective"] + AGENTS


def fmt(T, cols=("F1", "MCC", "ROC-AUC", "PR-AUC")):
    o = pd.DataFrame({"Model": T.Model, "Seeds": T.seeds})
    for c in cols:
        o[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(T[c], T[c + "_sd"])]
        if c + "_lo" in T:
            o[c + " 95% CI"] = [f"[{a:.2f}, {b:.2f}]" for a, b in zip(T[c + "_lo"], T[c + "_hi"])]
    for c in ("F1", "PR-AUC"):
        if f"dPR-AUC_lo" in T and f"pHolm_{c}" in T:
            o[f"Δ{c} vs AMIF-X [CI]"] = [("—" if pd.isna(a) else f"[{a:+.2f}, {b:+.2f}]") for a, b in zip(T[f"d{c}_lo"], T[f"d{c}_hi"])]
            o[f"p_Holm({c})"] = [("—" if pd.isna(p) else f"{p:.3g}") for p in T[f"pHolm_{c}"]]
    if "p_DeLong" in T:
        o["p_DeLong"] = [("—" if pd.isna(p) else f"{p:.3g}") for p in T["p_DeLong"]]
    return o


def groups_of(ds, split, scen):
    s = available(ds, split, scen)
    if not s:
        return []
    _, info = load_run(tag(ds, split, scen, s[0]))
    g = list(info["groups"])
    for sx in ("+base", "+base2", "+gnn", "+arf"):
        if available(ds, split, scen + sx):
            g += [x for x in load_run(tag(ds, split, scen + sx, available(ds, split, scen + sx)[0]))[1]["groups"] if x not in g]
    return list(dict.fromkeys([x for x in ORDER if x in g] + [x for x in g if x not in ORDER]))


def main_tables(ds):
    for scen in ["oracle", "static", "realistic", "budget10", "noise20", "delay2"]:
        G = groups_of(ds, "main", scen)
        if not G:
            continue
        cached = TABLES / f"{ds}_{scen}_full.csv"
        if __import__("os").environ.get("AMIFX_REUSE") and cached.exists():
            T = pd.read_csv(cached)
        else:
            T = summary_table(ds, "main", scen, G, B=B)
        T.to_csv(TABLES / f"{ds}_{scen}_full.csv", index=False)
        fmt(T).to_csv(TABLES / f"{ds}_{scen}.csv", index=False)
        NUM[f"{ds}_{scen}"] = T.set_index("Model")[["F1", "F1_sd", "PR-AUC", "PR-AUC_sd", "ROC-AUC", "MCC"]].round(2).to_dict("index")
        print(f"== {ds} {scen}\n", fmt(T)[["Model", "F1", "PR-AUC"] + [c for c in fmt(T).columns if "Holm(PR" in c]].to_string())


def decomposition(ds):
    rows = []
    for g in ["LightGBM", "LightGBM-AllViews", "LightGBM-Collective", "Stacking", "Stacking-AllViews", "AMIF-X"]:
        r = {"Model": g}
        for scen in ["static", "oracle", "realistic"]:
            A = group_arrays(ds, "main", scen, g) if available(ds, "main", scen) else None
            if A is None:
                continue
            ps = per_seed_metrics(*A[:3])
            r[f"{scen} PR-AUC"] = f"{ps['PR-AUC'].mean():.2f} ± {ps['PR-AUC'].std(ddof=1):.2f}"
            r[f"{scen} F1"] = f"{ps['F1'].mean():.2f} ± {ps['F1'].std(ddof=1):.2f}"
            NUM.setdefault(f"{ds}_decomp", {}).setdefault(g, {})[scen] = [ps["PR-AUC"].mean(), ps["F1"].mean()]
        rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / f"{ds}_decomposition.csv", index=False)


def sweep(ds, scens):
    rows = []
    models = ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM", "LightGBM-SelfTrain", "LightGBM-Decay", "XGBoost", "CatBoost"]
    for scen in scens:
        if not available(ds, "main", scen):
            continue
        R = group_arrays(ds, "main", scen, "AMIF-X")
        for g in models:
            A = group_arrays(ds, "main", scen, g)
            if A is None:
                continue
            ps = per_seed_metrics(*A[:3])
            r = {"Scenario": scen, "Model": g, "PR-AUC": ps["PR-AUC"].mean(), "PR-AUC_sd": ps["PR-AUC"].std(ddof=1),
                 "F1": ps["F1"].mean(), "F1_sd": ps["F1"].std(ddof=1)}
            if g != "AMIF-X":
                lo, hi, p = pooled_boot(A[0], R[1], R[2], "PR-AUC", A[3], B=B // 2, ref=(A[1], A[2]))
                r.update({"dPR_lo": lo, "dPR_hi": hi, "p": p})
            rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T.to_csv(TABLES / f"{ds}_feedback_sweep.csv", index=False)
        NUM[f"{ds}_sweep"] = T.round(3).to_dict("records")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for g in models:
            s = T[T.Model == g]
            if len(s):
                for k, m in enumerate(["PR-AUC", "F1"]):
                    ax[k].errorbar(range(len(s)), s[m], yerr=s[m + "_sd"], marker="o", ms=3, capsize=2, label=g,
                                   lw=2 if g == "AMIF-X" else 1)
        for k, m in enumerate(["PR-AUC", "F1"]):
            ax[k].set_xticks(range(len(T.Scenario.unique()))); ax[k].set_xticklabels(T.Scenario.unique(), rotation=40, ha="right")
            ax[k].set_ylabel(m + " (%)"); ax[k].grid(alpha=.3)
        ax[1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{ds}: detection under realistic analyst feedback (mean ± s.d. over seeds)")
        fig.savefig(FIGURES / f"{ds}_feedback_sweep.png"); plt.close(fig)


def ablations(ds, scen="oracle"):
    if not available(ds, "main", scen):
        return
    ref = replay_arrays(ds, "main", scen, AGENTS)
    configs = {"AMIF-X (learned planner + context)": dict(members=AGENTS)}
    configs["  w/o context"] = dict(members=AGENTS, context=False)
    configs["  + agent x context interactions"] = dict(members=AGENTS, interactions=True)
    configs["  equal-weight log-odds fusion (AMIF-style)"] = dict(members=AGENTS, kind="equal")
    configs["  mean-probability fusion"] = dict(members=AGENTS, kind="mean_prob")
    configs["  confidence-weighted fusion"] = dict(members=AGENTS, kind="confidence")
    for a in AGENTS:
        configs[f"  w/o {a}"] = dict(members=[m for m in AGENTS if m != a])
    configs["  FullProfile only"] = dict(members=["FullProfile"], context=False)
    configs["  FullProfile + context"] = dict(members=["FullProfile"])
    configs["  specialists only (w/o FullProfile)"] = dict(members=[m for m in AGENTS if m != "FullProfile"])
    rows = []
    for nm, kw in configs.items():
        mem = kw.pop("members")
        A = replay_arrays(ds, "main", scen, mem, **kw) if nm != "AMIF-X (learned planner + context)" else ref
        ps = per_seed_metrics(*A[:3])
        r = {"Configuration": nm, "F1": ps.F1.mean(), "F1_sd": ps.F1.std(ddof=1), "PR-AUC": ps["PR-AUC"].mean(),
             "PR-AUC_sd": ps["PR-AUC"].std(ddof=1), "ROC-AUC": ps["ROC-AUC"].mean()}
        if A is not ref:
            lo, hi, p = pooled_boot(A[0], A[1], A[2], "PR-AUC", A[3], B=B // 2, ref=(ref[1], ref[2]))
            r.update({"dPR-AUC": r["PR-AUC"] - per_seed_metrics(*ref[:3])["PR-AUC"].mean(), "dPR_lo": lo, "dPR_hi": hi, "p": p})
        rows.append(r)
    T = pd.DataFrame(rows)
    T["pHolm"] = np.nan
    m = T.p.notna(); T.loc[m, "pHolm"] = st.holm(T.loc[m, "p"].values)
    T.to_csv(TABLES / f"{ds}_{scen}_ablation.csv", index=False)
    NUM[f"{ds}_{scen}_ablation"] = T.round(3).to_dict("records")


def gating(ds):
    if not available(ds, "main", "oracle"):
        return
    rows = []
    full = replay_arrays(ds, "main", "oracle", AGENTS)
    ps = per_seed_metrics(*full[:3])
    rows.append({"Band": "all agents", "Activation": 100.0, "Agents/case": 8.0, "F1": ps.F1.mean(), "PR-AUC": ps["PR-AUC"].mean()})
    t1 = replay_arrays(ds, "main", "oracle", TIER1)
    ps = per_seed_metrics(*t1[:3])
    rows.append({"Band": "Tier-1 only", "Activation": 0.0, "Agents/case": 3.0, "F1": ps.F1.mean(), "PR-AUC": ps["PR-AUC"].mean()})
    for band in [(0.1, 0.8), (0.05, 0.9), (0.02, 0.95), (0.01, 0.99)]:
        A = replay_arrays(ds, "main", "oracle", AGENTS, tier1=TIER1, band=band)
        ps = per_seed_metrics(*A[:3]); act = np.mean([a.mean() for a in A[4]])
        rows.append({"Band": str(band), "Activation": 100 * act, "Agents/case": 3 + 5 * act, "F1": ps.F1.mean(),
                     "PR-AUC": ps["PR-AUC"].mean()})
    T = pd.DataFrame(rows)
    rt = RESULTS / f"{ds}_runtime.json"
    if rt.exists():
        R = json.loads(rt.read_text())
        T["Measured s/step"] = [R["full_seconds"], np.nan] + [R.get(f"gated{b}", {}).get("seconds", np.nan) for b in
                                                             [(0.1, 0.8), (0.05, 0.9), (0.02, 0.95), (0.01, 0.99)]]
    T.to_csv(TABLES / f"{ds}_gating.csv", index=False)
    NUM[f"{ds}_gating"] = T.round(3).to_dict("records")


def periods(ds):
    if ds != "elliptic" or not available(ds, "main", "oracle"):
        return
    rows = []
    for g in ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM", "CatBoost", "GCN", "LightGBM-Decay"]:
        A = group_arrays(ds, "main", "oracle", g)
        if A is None:
            continue
        y, S, Dd, stp = A
        for nm, m in [("pre-shutdown (36-42)", stp < SHUTDOWN_STEP), ("post-shutdown (43-49)", stp >= SHUTDOWN_STEP)]:
            ps = per_seed_metrics(y[m], S[:, m], Dd[:, m])
            lo, hi, _ = pooled_boot(y[m], S[:, m], Dd[:, m], "PR-AUC", stp[m], B=B // 2)
            rows.append({"Model": g, "Period": nm, "n_illicit": int(y[m].sum()), "F1": ps.F1.mean(), "PR-AUC": ps["PR-AUC"].mean(),
                         "PR-AUC_lo": lo, "PR-AUC_hi": hi, "ROC-AUC": ps["ROC-AUC"].mean()})
    T = pd.DataFrame(rows); T.to_csv(TABLES / "elliptic_periods.csv", index=False)
    NUM["elliptic_periods"] = T.round(3).to_dict("records")


def per_step_fig(ds):
    if not available(ds, "main", "oracle"):
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    for g in ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM", "GCN"]:
        A = group_arrays(ds, "main", "oracle", g)
        if A is None:
            continue
        y, S, Dd, stp = A
        us = np.unique(stp)
        f = [[st._prf(y[stp == s], d[stp == s])[2] * 100 for s in us] for d in Dd]
        ax.plot(us, np.mean(f, 0), marker="o", ms=3, label=g, lw=2 if g == "AMIF-X" else 1)
    if ds == "elliptic":
        ax.axvspan(SHUTDOWN_STEP - .5, us.max() + .5, color="grey", alpha=.15, label="after dark-market shutdown")
    ax.set_xlabel("time step"); ax.set_ylabel("illicit-class F1 (%)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.savefig(FIGURES / f"{ds}_per_step_f1.png"); plt.close(fig)


def pr_fig(ds, scen="oracle"):
    if not available(ds, "main", scen):
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for g in ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM", "GCN"]:
        A = group_arrays(ds, "main", scen, g)
        if A is None:
            continue
        y, S = A[0], A[1]
        p, r, _ = precision_recall_curve(y, S[0])
        ax.plot(r, p, label=f"{g} ({st.metrics(y, S[0], .5)['PR-AUC']:.1f})", lw=2 if g == "AMIF-X" else 1)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title(f"{ds} ({scen}, seed 0; legend: PR-AUC %)", fontsize=8)
    fig.savefig(FIGURES / f"{ds}_{scen}_pr.png"); plt.close(fig)


def weights_fig(ds):
    s = available(ds, "main", "oracle")
    if not s:
        return
    W = []
    for seed in s:
        _, info = load_run(tag(ds, "main", "oracle", seed))
        c = info["coefs"]["AMIF-X"]
        W.append(pd.DataFrame({int(t): v["beta"] for t, v in c.items()}).T)
    M = sum(W) / len(W); SD = (sum((w - M) ** 2 for w in W) / max(len(W) - 1, 1)) ** .5
    fig, ax = plt.subplots(figsize=(7, 3))
    for a in AGENTS:
        ax.plot(M.index, M[a], marker=".", label=a); ax.fill_between(M.index, M[a] - SD[a], M[a] + SD[a], alpha=.15)
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("refit step"); ax.set_ylabel("planner weight β (log-odds)")
    ax.legend(fontsize=6, ncol=4); ax.grid(alpha=.3)
    fig.savefig(FIGURES / f"{ds}_planner_weights.png"); plt.close(fig)
    M.round(3).to_csv(TABLES / f"{ds}_planner_weights.csv")


def budget(ds):
    if not available(ds, "main", "oracle"):
        return
    rows = []
    for g in ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM", "CatBoost"]:
        A = group_arrays(ds, "main", "oracle", g)
        if A is None:
            continue
        for f in [0.01, 0.02, 0.05, 0.10, 0.20]:
            k = [st.topk(A[0], s, f) for s in A[1]]
            rows.append({"Model": g, "Budget": f"{f*100:g}%", "Precision@K": np.mean([x["Precision@K"] for x in k]),
                         "Recall@K": np.mean([x["Recall@K"] for x in k]), "Lift@K": np.mean([x["Lift@K"] for x in k])})
    T = pd.DataFrame(rows); T.to_csv(TABLES / f"{ds}_budget.csv", index=False)
    NUM[f"{ds}_budget"] = T.round(2).to_dict("records")


def cvcl_ablation(ds):
    rows, qual = [], []
    for scen in ["realistic", "budget10", "noise20", "delay2", "oracle"]:
        if not available(ds, "main", scen):
            continue
        R = group_arrays(ds, "main", scen, "AMIF-X")
        for var, nm in [("", "AMIF-X (final configuration)"), ("+cvcl", "  + cross-view consensus learning (CVCL)"),
                        ("-symfilter+cvcl", "  + CVCL with symmetric noise filter")]:
            if var and not available(ds, "main", scen + var):
                continue
            A = R if not var else group_arrays(ds, "main", scen + var, "AMIF-X")
            ps = per_seed_metrics(*A[:3])
            r = {"Scenario": scen, "Variant": nm, "seeds": len(ps), "PR-AUC": ps["PR-AUC"].mean(), "PR-AUC_sd": ps["PR-AUC"].std(ddof=1),
                 "F1": ps.F1.mean(), "F1_sd": ps.F1.std(ddof=1)}
            if var:
                n = min(len(A[1]), len(R[1]))
                lo, hi, p = pooled_boot(A[0], R[1][:n], R[2][:n], "PR-AUC", A[3], B=B // 2, ref=(A[1][:n], A[2][:n]))
                r.update({"full_minus_variant_lo": lo, "full_minus_variant_hi": hi, "p": p})
            rows.append(r)
        for s in available(ds, "main", scen + "+cvcl"):
            _, info = load_run(tag(ds, "main", scen + "+cvcl", s))
            for o, d in info.get("cvcl", {}).items():
                if not d:
                    continue
                last = d[max(d, key=int)]
                tot_p = sum(v["pseudo"] for v in d.values()); tot_w = sum(v["pseudo_wrong"] for v in d.values())
                tot_d = sum(v["dropped"] for v in d.values()); tot_dc = sum(v["dropped_correct"] for v in d.values())
                qual.append({"Scenario": scen, "seed": s, "Owner": o, "pseudo_labels(final step)": last["pseudo"],
                             "pseudo_accuracy(all refits)": 100 * (1 - tot_w / tot_p) if tot_p else np.nan,
                             "dropped(final step)": last["dropped"],
                             "dropped_that_were_noisy(%)": 100 * (1 - tot_dc / tot_d) if tot_d else np.nan})
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / f"{ds}_cvcl_ablation.csv", index=False)
        NUM[f"{ds}_cvcl_ablation"] = pd.DataFrame(rows).round(3).to_dict("records")
    if qual:
        Q = pd.DataFrame(qual).groupby(["Scenario", "Owner"]).mean(numeric_only=True).drop(columns="seed").round(2).reset_index()
        Q.to_csv(TABLES / f"{ds}_cvcl_quality.csv", index=False); NUM[f"{ds}_cvcl_quality"] = Q.to_dict("records")


def splits(ds):
    rows = []
    for sp in ["main", "early", "late"]:
        for scen in ["oracle", "realistic"]:
            if not available(ds, sp, scen):
                continue
            for g in ["AMIF-X", "LightGBM-Collective", "Stacking", "LightGBM-AllViews", "LightGBM"]:
                A = group_arrays(ds, sp, scen, g)
                if A is None:
                    continue
                ps = per_seed_metrics(*A[:3])
                rows.append({"Split": sp, "Scenario": scen, "Model": g, "seeds": len(ps), "PR-AUC": ps["PR-AUC"].mean(),
                             "PR-AUC_sd": ps["PR-AUC"].std(ddof=1), "F1": ps.F1.mean(), "F1_sd": ps.F1.std(ddof=1)})
    if rows:
        T = pd.DataFrame(rows); T.to_csv(TABLES / f"{ds}_splits.csv", index=False)
        NUM[f"{ds}_splits"] = T.round(3).to_dict("records")


if __name__ == "__main__":
    for ds in ["elliptic", "amlsim10k", "amlsim100k"]:
        print("#####", ds, flush=True)
        main_tables(ds); decomposition(ds)
        sweep(ds, ["oracle", "delay1", "delay2", "delay4", "budget20", "budget10", "budget5", "noise10", "noise20",
                   "noise30", "realistic"])
        cvcl_ablation(ds); ablations(ds); gating(ds); periods(ds); per_step_fig(ds); pr_fig(ds); weights_fig(ds); budget(ds); splits(ds)
        (RESULTS / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print("done")
