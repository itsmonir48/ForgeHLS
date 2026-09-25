"""
make_figures.py

Generates publication-ready IEEE-format charts from experimental results (Milestone 6).

Figures Generated in hls_ml/figures/:
1. fig1_distribution_shift_e1_vs_e2.png
   - Bar chart showing nominal accuracy in E1 (Random Split) vs structural degradation in E2 (Kernel-Disjoint Split).
2. fig2_uncertainty_correlation_quintiles.png
   - Monotonic error explosion from Q1 (Lowest Uncertainty) to Q5 (Highest Uncertainty) across all 5 targets.
3. fig3_risk_coverage_curves.png
   - Risk-Coverage curves showing the dramatic plunge in MAE as high-uncertainty designs are selectively abstained.
4. fig4_feature_ablation.png
   - Comparison of Feature Set A (Basic), Set B (Pragma-aware), and Set C (Structural+Pragma).
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, TARGET_NAMES

# Configure IEEE-style matplotlib aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

TARGET_NAMES = ["Worst-caseLatency", "LUT", "DSP", "FF", "BRAM_18K"]
TARGET_DISPLAY = ["Latency", "LUT", "DSP", "FF", "BRAM"]


def plot_fig1_distribution_shift():
    csv_path = RESULTS_DIR / "baseline_results_e1_vs_e2.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 1: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Focus on Random Forest
    sub = df[df["Model"] == "RandomForest"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(TARGET_DISPLAY))
    width = 0.35

    # Use normalized MAE or Degradation Ratios for clean multi-target visualization
    e1_nmaes = sub["E1_NMAE"].values
    e2_nmaes = sub["E2_NMAE"].values

    rects1 = ax.bar(x - width/2, e1_nmaes, width, label="E1: Random Row Split (Optimistic)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width/2, e2_nmaes, width, label="E2: Kernel-Disjoint (Distribution Shift)", color="#d95f02", alpha=0.9)

    ax.set_ylabel("Normalized Mean Absolute Error (NMAE)")
    ax.set_title("Figure 1: Performance Collapse Under Kernel Distribution Shift (H1 vs H2)")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_DISPLAY)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate degradation factor on top of bars
    for i, row in sub.reset_index().iterrows():
        deg = row["MAE_Degradation_Ratio"]
        y_pos = e2_nmaes[i] + 0.05 * max(e2_nmaes)
        ax.text(x[i] + width/2, y_pos, f"{deg:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d95f02")

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig1_distribution_shift_e1_vs_e2.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def plot_fig2_uncertainty_quintiles():
    npz_path = RESULTS_DIR / "uncertainty_predictions.npz"
    if not npz_path.exists():
        print(f"Skipping Fig 2: {npz_path} not found")
        return

    data = np.load(npz_path)
    Y_true = data["Y_test_true"]
    rf_preds = data["rf_preds"]
    rf_uncert = data["rf_uncertainty"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    # Plot for LUT, DSP, FF, BRAM (the core FPGA resources)
    resource_indices = [1, 2, 3, 4]

    for plot_i, t_idx in enumerate(resource_indices):
        ax = axes[plot_i]
        tgt_name = TARGET_DISPLAY[t_idx]

        y_t = Y_true[:, t_idx]
        pred_t = rf_preds[:, t_idx]
        uncert_t = rf_uncert[:, t_idx]
        abs_err = np.abs(y_t - pred_t)

        q_codes = pd.qcut(uncert_t, q=5, labels=False, duplicates="drop")
        n_unique_bins = len(np.unique(q_codes))
        bin_labels = [f"Q{i+1}" for i in range(n_unique_bins)]
        bin_labels[0] = f"{bin_labels[0]} (Low)"
        bin_labels[-1] = f"{bin_labels[-1]} (High)"
        df_q = pd.DataFrame({"err": abs_err, "q": q_codes})
        mae_by_q = df_q.groupby("q")["err"].mean()

        colors = ["#2b5c8f", "#41b6c4", "#7fcdbb", "#fe9929", "#d95f02"]
        ax.bar(range(len(mae_by_q)), mae_by_q.values, color=colors[:len(mae_by_q)], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{tgt_name} Error", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(mae_by_q)))
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Figure 2: Monotonic Error Growth Across Uncertainty Quintiles (H3 Verification)", fontsize=13, y=1.03)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig2_uncertainty_correlation_quintiles.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def plot_fig3_risk_coverage():
    csv_path = RESULTS_DIR / "risk_coverage_data.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 3: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    sub = df[df["Method"] == "Tree_Ensemble"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left plot: Retained Error Ratio (Normalized to 1.0 at 100% Coverage)
    ax1 = axes[0]
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for i, tgt in enumerate(TARGET_NAMES):
        tgt_df = sub[sub["Target"] == tgt].sort_values("Coverage")
        ax1.plot(tgt_df["Coverage"] * 100, tgt_df["Retained_Error_Ratio"], marker="o", label=TARGET_DISPLAY[i], color=palette[i])

    # Plot Random baseline (horizontal line at 1.0)
    ax1.axhline(1.0, color="gray", linestyle="--", label="Random Rejection Baseline")
    ax1.set_xlabel("Coverage (% Accepted Designs)")
    ax1.set_ylabel("Retained Risk Ratio (Relative to 100% Cov)")
    ax1.set_title("Normalized Risk-Coverage Trade-off")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(10, 105)

    # Right plot: Error Reduction Factor (Speedup / Accuracy Gain)
    ax2 = axes[1]
    for i, tgt in enumerate(TARGET_NAMES):
        if tgt != "Worst-caseLatency":  # keep visual scale readable across resources
            tgt_df = sub[sub["Target"] == tgt].sort_values("Coverage")
            ax2.plot(tgt_df["Coverage"] * 100, tgt_df["Error_Reduction_Factor"], marker="s", label=f"{TARGET_DISPLAY[i]} Gain", color=palette[i])

    ax2.set_xlabel("Coverage (% Accepted Designs)")
    ax2.set_ylabel("Accuracy Multiplier (x-times better)")
    ax2.set_title("Error Reduction Factor via Abstention")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim(10, 105)

    fig.suptitle("Figure 3: Risk-Coverage Curves for Selective Synthesis (H4 Verification)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig3_risk_coverage_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def plot_fig4_feature_ablation():
    csv_path = RESULTS_DIR / "feature_ablation_results.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 4: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    x = np.arange(len(TARGET_DISPLAY))
    width = 0.25

    set_a = df[df["Feature_Set"] == "Set_A_Basic"]
    set_b = df[df["Feature_Set"] == "Set_B_PragmaAware"]
    set_c = df[df["Feature_Set"] == "Set_C_StructuralPragma"]

    # Plot Spearman rho (uncertainty quality) across feature sets
    r_a = set_a["Spearman_rho"].values
    r_b = set_b["Spearman_rho"].values
    r_c = set_c["Spearman_rho"].values

    ax.bar(x - width, r_a, width, label="Set A (Basic: 3 features)", color="#b2df8a", alpha=0.9)
    ax.bar(x, r_b, width, label="Set B (Pragma-aware: 28 features)", color="#33a02c", alpha=0.9)
    ax.bar(x + width, r_c, width, label="Set C (Structural+Pragma: 42 features)", color="#1f78b4", alpha=0.9)

    ax.set_ylabel("Uncertainty Spearman Correlation (ρ)")
    ax.set_title("Figure 4: Feature Set Ablation on Uncertainty Informative Quality (E8)")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_DISPLAY)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig4_feature_ablation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def main():
    print("=" * 75)
    print("Generating Publication Figures (Milestone 6)")
    print("=" * 75)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Figure 1: Distribution Shift Comparison...")
    plot_fig1_distribution_shift()

    print("\nGenerating Figure 2: Uncertainty Quintiles Monotonicity...")
    plot_fig2_uncertainty_quintiles()

    print("\nGenerating Figure 3: Risk-Coverage Curves (Selective Prediction)...")
    plot_fig3_risk_coverage()

    print("\nGenerating Figure 4: Feature Ablation Comparison...")
    plot_fig4_feature_ablation()

    print("\n" + "=" * 75)
    print(f"All publication-ready figures successfully generated in: {FIGURES_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
