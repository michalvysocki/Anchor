"""
Figures of e02 for the paper, from the saved evaluation and predictions
(no training): trained vs predicted gains, and the critic vs the raw
residual. Written to paper/figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anchor.plotting import VARIANT_COLORS

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR.parents[1] / "paper" / "figures"
COLORS = {
    "C": VARIANT_COLORS["C"],
    "D": VARIANT_COLORS["D"],
    "M": "#7f7f7f",
    "M*": "#000000",
}
MARKERS = {"C": "o", "D": "s", "M": "^", "M*": "x"}
# placement of Markov configurations (zero gap) on the log axis
MARKOV_GAP = 1e-6

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def main():
    results = json.loads((RESULTS_DIR / "test_results.json").read_text())["results"]
    predictions = json.loads((RESULTS_DIR / "predictions.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 2.4))

    for variant in ("C", "D", "M", "M*"):
        xs, ys, es = [], [], []
        for name, r in results.items():
            if not name.endswith("_large") or r["markov"]:
                continue
            d = r["variants"][variant]
            p = d["per_lambda"][str(d["lambda"])]
            xs.append(p["predicted"])
            ys.append(p["log_ratio"][0])
            es.append(p["log_ratio"][1])
        left.errorbar(
            xs,
            ys,
            yerr=es,
            fmt=MARKERS[variant],
            ms=3.5,
            mfc="none" if variant != "M*" else None,
            color=COLORS[variant],
            lw=0.8,
            label=variant,
        )
    lim = (-0.38, 0.02)
    left.plot(lim, lim, "--", color="#aaaaaa", lw=0.8, zorder=0)
    left.set_xlim(lim)
    left.set_ylim(lim)
    left.set_xlabel("predicted log(MSE / one-step), population")
    left.set_ylabel("trained, mean ± 95% CI (30 seeds)")
    left.set_title("(a) Theory vs training")
    left.legend(frameon=False, loc="upper left")

    for name, r in results.items():
        if not name.endswith("_large"):
            continue
        p = predictions[name.removesuffix("_large")]
        gap = max(np.mean(p["onestep"]) - np.mean(p["direct"]), MARKOV_GAP)
        right.scatter(gap, r["critic"], s=10, color="black", marker="o")
        right.scatter(gap, r["residual"], s=10, color="#999999", marker="x")
    right.scatter([], [], s=10, color="black", marker="o", label="critic")
    right.scatter([], [], s=10, color="#999999", marker="x", label="raw residual")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Markov gap: one-step − direct MSE (Markov at $10^{-6}$)")
    right.set_ylabel("estimate at the trained one-step model")
    right.set_title("(b) The critic measures the gap")
    right.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "e02_theory_vs_training.pdf")


if __name__ == "__main__":
    main()
