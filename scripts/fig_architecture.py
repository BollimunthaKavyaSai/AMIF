"""Architecture diagram of AMIF-X with Cross-View Consensus Learning (results/figures/architecture.png)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from amifx.config import FIGURES  # noqa: E402

fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(-1, 122); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff", ec="#333333", fs=8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(a, b, c="#333333", ls="-", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# evidence views
ax.text(9, 63, "Point-in-time evidence views (step t)", ha="center", fontsize=9, weight="bold")
views = [("tx", "Transaction behaviour"), ("nb", "Neighbourhood aggregates"), ("tp", "Temporal context (within-step)"),
         ("ac", "Actor / wallet history (≤ t)"), ("gs", "Graph structure (within-step)"), ("ctx", "Case context z")]
for i, (k, t) in enumerate(views):
    box(0, 54 - i * 8.2, 20, 6, f"{k}: {t}", fc="#eef3fb", fs=6.6)

# agents
ax.text(37, 63, "Specialist agents", ha="center", fontsize=9, weight="bold")
t1 = ["FullProfile (tx∪nb)", "Transaction (tx)", "Temporal (tp)"]
t2 = ["Neighbourhood (nb)", "Actor (ac)", "Graph – decoupled SAGE", "Anomaly – isolation forest", "NeighbourRisk – propagation"]
for i, a in enumerate(t1):
    box(28, 55 - i * 5.6, 18, 4.3, a, fc="#e8f5e9", fs=7)
ax.text(37, 38.6, "Tier 1 (always)", ha="center", fontsize=7, style="italic")
for i, a in enumerate(t2):
    box(28, 32 - i * 5.6, 18, 4.3, a, fc="#fff3e0", fs=7)
ax.text(37, 6.3, "Tier 2 (only if Tier-1 risk is uncertain)", ha="center", fontsize=7, style="italic")
for i in range(6):
    arrow((20.6, 57 - i * 8.2), (27.4, 45 if i < 3 else 22), c="#7a8ba6", lw=0.7)

# planner
box(52, 34, 19, 14, "Stacked planner\nR = σ(b + Σ βⱼ ℓ̃ⱼ + γᵀz̃)\nfitted on out-of-sample\nhistory only", fc="#ede7f6", fs=7.5, bold=False)
box(52, 18, 19, 11, "Two-tier gate\nrun Tier 2 iff\nl ≤ R⁽¹⁾ ≤ u", fc="#f3e5f5", fs=7.5)
arrow((46.5, 50), (51.6, 43)); arrow((46.5, 22), (51.6, 38)); arrow((61.5, 29.4), (61.5, 33.6))

# decision + explanation
box(77, 40, 19, 10, "Decision R ≥ θₜ\n(θ tuned on own history)\n+ investigation ranking", fc="#ffffff", fs=7.5)
box(77, 26, 19, 11, "Explanation\nφⱼ = βⱼ(ℓⱼ − E[ℓⱼ])\n= exact Shapley values\n+ view-level Shapley", fc="#ffffff", fs=7.5)
arrow((71.5, 43), (76.6, 45)); arrow((71.5, 38), (76.6, 32))

# analysts
box(101, 36, 18, 14, "Analysts\nreview budget b + audit\ndelay d · error ε_FN, ε_FP", fc="#fdecea", fs=7.5, bold=False)
arrow((96.5, 45), (100.6, 44))

# CVCL
box(74, 2, 47, 18, "", fc="#e0f7fa", ec="#00838f")
ax.text(97.5, 17.6, "Cross-View Consensus Learning (CVCL)", ha="center", fontsize=8.5, weight="bold", color="#006064")
ax.text(75.5, 13.8, "(i)  pseudo-label unreviewed / delayed cases when the views agree", fontsize=6.9)
ax.text(75.5, 10.3, "(ii) drop analyst labels that all views strongly contradict", fontsize=6.9)
ax.text(75.5, 6.8, "(iii) spend 20% of the review budget on maximal view disagreement", fontsize=6.9)
arrow((110, 35.6), (110, 20.4), c="#00838f")
arrow((75.6, 11), (46.8, 4.0), c="#00838f", ls="--", rad=-0.15)
ax.text(58, 2.0, "refit all agents at t+1 on recorded + consensus labels", fontsize=7, color="#006064", ha="center")
arrow((110, 50.6), (61.5, 48.6), c="#c62828", ls=":", rad=0.35)
ax.text(86, 60.5, "recorded labels → planner & threshold refit", fontsize=7, color="#c62828", ha="center")
fig.savefig(FIGURES / "architecture.png", dpi=200, bbox_inches="tight")
print("saved")
