# AMIF-X — evidence-view agents with relational risk propagation under realistic analyst feedback

Reproducible code and results for *"AMIF-X: Evidence-View Agents with Relational Risk Propagation for
Financial-Crime Investigation under Realistic Analyst Feedback"*. Every result table and figure is produced by the
scripts below from the raw data. (The earlier code in `../AIMF_Project/AMIFX` is superseded and was not reused.)

## Final model (selected on development splits only)
* Eight agents, all LightGBM except the isolation-forest Anomaly agent, each reading one point-in-time evidence view:
  FullProfile (tx∪nb), Transaction, Neighbourhood, Temporal, Actor, Graph (two-hop propagated features),
  Anomaly, NeighbourRisk (risk of counterparties from an all-view first stage, out-of-fold).
* Stacked logistic planner on out-of-sample agent log-odds + case context; centred contributions = exact Shapley values.
* Threshold: F1-optimal on each model's own out-of-sample history (all models, no fixed 0.5).
* Cross-View Consensus Learning (CVCL) is an evaluated extension (`--cvcl`), not part of the final model.
* Development record: `scripts/dev*.py`, `results/tables/dev_*.csv` (variants v0–v17; v15 selected by the rule
  "highest mean development PR-AUC across data sets; differences < 0.5 = tie → simpler").

## Reviewer concern → where it is addressed
| # | Concern | What the code does |
|---|---|---|
| 1 | Gain from feedback, not architecture | Static / oracle / realistic protocols × all models; single models also on all views; collective two-stage baseline; fusion and agent ablations (`analyse.py`) |
| 2 | Unrealistic human-in-the-loop | Feedback simulator: delay, budgeted review + random audit, asymmetric analyst error (`config.py::FEEDBACK`, `protocol.py`) |
| 3 | Unfair / missing baselines | History-tuned thresholds for every model; stacking, collective, GCN/GraphSAGE/GAT/SGC, drift-aware and self-training baselines (`models/learners.py`) |
| 4 | Leakage | Per-refit fitting only; within-step graph; `scripts/leakage_test.py` rebuilds features after deleting all future data (bit-identical) |
| 5 | Weak statistics | 5 seeds; stratified bootstrap CIs; paired bootstrap tests; DeLong; Wilcoxon; Holm (`stats.py`, `evaluate.py`) |
| 6 | Explanations | Exactness, faithfulness under refit, view-level Shapley vs. TreeSHAP, stability (`explain.py`) |
| 7 | One data set | AMLSim money-laundering simulation added (`data/amlsim.py`) |
| Ed. | Efficiency | Measured wall-clock time of gated inference (`runtime.py`); fault tolerance under evidence outages (`outage.py`) |

## Reproduce
```
pip install -r requirements.txt
python -m amifx.data.elliptic            # Elliptic++ csvs in AIMF_Project/primary_dataset (or set AMIFX_ELLIPTIC)
python -m amifx.data.amlsim amlsim10k    # AMLSim 10K csvs (or set AMIFX_AMLSIM10K)
python scripts/leakage_test.py
python scripts/jobs.py main              # all prequential runs; resumable; AMIFX_WORKERS / AMIFX_THREADS
python scripts/explain_runtime.py        # explanation validation, outage test, measured runtime
python scripts/analyse.py                # tables, figures, results/numbers.json
```
Hardware used: 12-thread CPU, 7.7 GB RAM, no GPU (one job at a time; the full grid takes about two days).

## Layout
```
amifx/config.py            splits (main, dev), seeds, feedback scenarios
amifx/data/                point-in-time case tables (Elliptic++, AMLSim)
amifx/models/learners.py   agents and all baselines behind one fit/predict interface
amifx/planner.py           stacked planner, centred Shapley contributions, threshold, two-tier gate
amifx/protocol.py          prequential protocol, analyst-feedback simulator, CVCL extension
amifx/evaluate.py, stats.py  metrics, bootstrap CIs, paired tests, fusion replays
amifx/explain.py, runtime.py, outage.py   validation experiments
scripts/                   job grid, development batches, analysis
results/                   predictions/, tables/, figures/, numbers.json
```

## What is in this repository
All code, every result table (`results/tables/`), figure (`results/figures/`), the explanation/runtime/leakage results (`results/*.json`) and the development-split record.
Per-run prediction files (`results/predictions/`, ~1.2 GB) and the feature cache (`cache/`) are not versioned because of
their size; they are regenerated exactly by the commands above (fixed seeds).
