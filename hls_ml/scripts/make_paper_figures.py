#!/usr/bin/env python3
"""
make_paper_figures.py -- the four figures used in the paper (no in-figure titles; captions live in the paper).

Needs: results/extra/robustness_raw.csv and results/extra/quintile data (run extra_analysis.py first),
       results/uncertainty_predictions.npz, results/feature_ablation_results.csv, data/*.npz
Writes: figures/paper_fig1.png ... paper_fig4.png   (sizes match IEEE single / double column widths)

  Fig 1  GB LogMAE: random rows (E1) vs kernel-disjoint (E3) vs constant predictor, five seeds
  Fig 2  mean log-scale error across tree-uncertainty quintiles, five targets (E2 test set)
  Fig 3  risk-coverage curves for latency and LUT (tree log-sigma, tree raw-sigma, |y_hat|, deep log-sigma, random)
  Fig 4  uncertainty-error Spearman rho for feature sets A/B/C
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, EXTRA, FIG, DATA = ROOT / "results", ROOT / "results" / "extra", ROOT / "figures", ROOT / "data"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
T = ["Latency", "LUT", "DSP", "FF", "BRAM"]
C_E1, C_E3, C_K = "#2b5c8f", "#d95f02", "#9a9a9a"


def fig1():
    raw = pd.read_csv(EXTRA / "robustness_raw.csv")
    a = raw[raw.exp.str.startswith("A")]; b = raw[raw.exp.str.startswith("B")]
    m = lambda d, c: [d[d.target == t][c].mean() for t in T]
    s = lambda d, c: [d[d.target == t][c].std() for t in T]
    x, w = np.arange(5), 0.26
    fig, ax = plt.subplots(figsize=(3.44, 1.98))
    ax.bar(x - w, m(a, "LogMAE"), w, yerr=s(a, "LogMAE"), color=C_E1, label="E1 random rows", capsize=1.5, error_kw={"lw": .6})
    ax.bar(x, m(b, "LogMAE"), w, yerr=s(b, "LogMAE"), color=C_E3, label="E3 kernel-disjoint", capsize=1.5, error_kw={"lw": .6})
    ax.bar(x + w, m(b, "const_LogMAE"), w, color=C_K, label="Constant (E3)")
    ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel("LogMAE (lower is better)")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", lw=.3, alpha=.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "paper_fig1.png"); plt.close(fig)


def load_unc():
    d = np.load(RES / "uncertainty_predictions.npz")
    return d, d["Y_test_true"]


def fig2():
    d, Y = load_unc()
    fig, axs = plt.subplots(1, 5, figsize=(7.08, 2.01))
    for j, (t, ax) in enumerate(zip(T, axs)):
        lerr = np.abs(np.log1p(Y[:, j]) - np.log1p(d["rf_preds"][:, j]))
        q = pd.qcut(d["rf_uncertainty"][:, j], 5, labels=False, duplicates="drop")
        means = [lerr[q == k].mean() for k in range(int(q.max()) + 1)]
        ax.bar(range(len(means)), means, color="#4c8c6b", width=.7)
        ax.set_xticks(range(len(means))); ax.set_xticklabels([f"Q{k+1}" for k in range(len(means))])
        ax.set_title(t); ax.grid(axis="y", lw=.3, alpha=.6); ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Mean log-scale error")
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "paper_fig2.png"); plt.close(fig)


def risk_curve(err, score, covs):
    o = np.argsort(score, kind="stable"); n = len(err)
    return np.array([err[o[:max(1, int(np.ceil(c * n)))]].mean() for c in covs])


def fig3():
    d, Y = load_unc()
    covs = np.linspace(0.05, 1.0, 20)
    fig, axs = plt.subplots(1, 2, figsize=(6.25, 2.66), sharey=True)
    for ax, j in zip(axs, [0, 1]):
        err = np.abs(Y[:, j] - d["rf_preds"][:, j]); full = err.mean()
        lines = [("Tree log σ", d["rf_uncertainty_log"][:, j], "-", "#2b5c8f"),
                 ("Tree raw σ", d["rf_uncertainty"][:, j], "-", "#4c8c6b"),
                 ("Predicted |ŷ|", d["rf_preds"][:, j], "--", "#d95f02"),
                 ("Deep log σ", d["deep_ensemble_uncertainty_log"][:, j], "-", "#8e44ad")]
        for lab, sc, ls, col in lines:
            ax.plot(covs * 100, np.maximum(risk_curve(err, sc, covs) / full, 1e-5), ls, color=col, label=lab)
        ax.plot(covs * 100, np.ones_like(covs), ":", color="k", label="Random rejection")
        ax.set_title(T[j]); ax.set_xlabel("Coverage (% accepted designs)"); ax.grid(lw=.3, alpha=.6)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 10)
    axs[0].set_ylabel("Retained MAE / MAE at 100% coverage")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "paper_fig3.png"); plt.close(fig)


def fig4():
    f = pd.read_csv(RES / "feature_ablation_results.csv")
    sets = [("Set_A_Basic", "Set A (3)", "#b7d7a8"), ("Set_B_PragmaAware", "Set B (28)", "#4c8c6b"),
            ("Set_C_StructuralPragma", "Set C (42)", "#2b5c8f")]
    x, w = np.arange(5), 0.26
    fig, ax = plt.subplots(figsize=(3.44, 1.82))
    for k, (key, lab, col) in enumerate(sets):
        v = f[f.Feature_Set == key]["Spearman_rho"].values
        ax.bar(x + (k - 1) * w, v, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel("Uncertainty–error ρ (log σ)")
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, ncol=3, loc="upper left"); ax.grid(axis="y", lw=.3, alpha=.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "paper_fig4.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written to", FIG)
