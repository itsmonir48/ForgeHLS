"""
make_paper_figures.py

Regenerates the four figures used in the paper from result files:
  Fig. 1  GB LogMAE, E1 random rows vs E3 kernel-disjoint (5 seeds) with the
          constant predictor            <- results/extra_e1_e3_summary.csv
  Fig. 2  mean log-scale error per tree-uncertainty quintile (E2 test)
                                        <- results/extra_uncertainty_quintiles.csv
  Fig. 3  risk-coverage curves for latency and LUT
                                        <- results/extra_risk_coverage_curves.csv
  Fig. 4  uncertainty-error rho (log sigma) per feature set, mean over the
          five E3 seeds with std error bars
                                        <- results/feature_ablation_e3_summary.csv

Run extra_analysis.py and ablation_study.py first. Output: hls_ml/figures/paper_fig*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR, FIGURES_DIR, TARGET_NAMES

DISPLAY = ["Latency", "LUT", "DSP", "FF", "BRAM"]
BLUE, ORANGE, GREY, GREEN, LIGHT, PURPLE = "#2b5c8f", "#d95f02", "#9a9a9a", "#4a8c6a", "#b8d8a8", "#8e44ad"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig1():
    df = pd.read_csv(RESULTS_DIR / "extra_e1_e3_summary.csv").set_index("Target").loc[TARGET_NAMES]
    x = np.arange(len(TARGET_NAMES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.bar(x - w, df["E1_LogMAE_mean"], w, yerr=df["E1_LogMAE_std"], capsize=2, color=BLUE, label="E1 random rows")
    ax.bar(x, df["E3_LogMAE_mean"], w, yerr=df["E3_LogMAE_std"], capsize=2, color=ORANGE, label="E3 kernel-disjoint")
    ax.bar(x + w, df["E3_Const_LogMAE_mean"], w, color=GREY, label="Constant (E3)")
    ax.set_xticks(x, DISPLAY)
    ax.set_ylabel("LogMAE (lower is better)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES_DIR / "paper_fig1_e1_vs_e3_logmae.png")
    plt.close(fig)


def fig2():
    df = pd.read_csv(RESULTS_DIR / "extra_uncertainty_quintiles.csv")
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.9))
    for ax, name, disp in zip(axes, TARGET_NAMES, DISPLAY):
        sub = df[df["Target"] == name].sort_values("Quintile_Bin")
        labels = [f"Q{i + 1}" for i in range(len(sub))]
        ax.bar(labels, sub["Mean_Log_Error"], color=GREEN, width=0.7)
        ax.set_title(disp, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Mean log-scale error")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGURES_DIR / "paper_fig2_uncertainty_quintiles.png")
    plt.close(fig)


def fig3():
    df = pd.read_csv(RESULTS_DIR / "extra_risk_coverage_curves.csv")
    styles = {
        "Tree_log_sigma": ("Tree log σ", BLUE, "-"),
        "Tree_raw_sigma": ("Tree raw σ", GREEN, "-"),
        "Magnitude": ("Predicted |ŷ|", ORANGE, "--"),
        "Deep_log_sigma": ("Deep log σ", PURPLE, "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for ax, name, disp in zip(axes, ["Worst-caseLatency", "LUT"], ["Latency", "LUT"]):
        for method, (label, color, ls) in styles.items():
            sub = df[(df["Target"] == name) & (df["Method"] == method)].sort_values("Coverage")
            ax.plot(sub["Coverage"] * 100, np.clip(sub["Retained_MAE_Ratio"], 1e-5, None),
                    color=color, ls=ls, lw=1.4, label=label)
        ax.axhline(1.0, color="black", ls=":", lw=1.2, label="Random rejection")
        ax.set_yscale("log")
        ax.set_title(disp, fontsize=10)
        ax.set_xlabel("Coverage (% accepted designs)")
        ax.grid(True, which="major", axis="both")
    axes[0].set_ylabel("Retained MAE / MAE at 100%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIGURES_DIR / "paper_fig3_risk_coverage.png")
    plt.close(fig)


def fig4():
    df = pd.read_csv(RESULTS_DIR / "feature_ablation_e3_summary.csv")
    sets = [("Set_A_Basic", "Set A (3)", LIGHT), ("Set_B_PragmaAware", "Set B (28)", GREEN),
            ("Set_C_StructuralPragma", "Set C (42)", BLUE)]
    x = np.arange(len(TARGET_NAMES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for k, (key, label, color) in enumerate(sets):
        sub = df[df["Feature_Set"] == key].set_index("Target").loc[TARGET_NAMES]
        ax.bar(x + (k - 1) * w, sub["Spearman_rho_mean"], w, yerr=sub["Spearman_rho_std"],
               capsize=2, color=color, label=label)
    ax.set_xticks(x, DISPLAY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Uncertainty–error ρ (log σ)")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.savefig(FIGURES_DIR / "paper_fig4_feature_ablation.png")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for make in (fig1, fig2, fig3, fig4):
        make()
        print(f"{make.__name__} written")


if __name__ == "__main__":
    main()
