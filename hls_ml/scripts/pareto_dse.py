"""
pareto_dse.py

Budgeted design-space exploration proxy on unseen kernels.

For every kernel-disjoint E3 split, GB (same configuration as train_baselines.py)
predicts all five targets for the held-out kernels. Inside each held-out kernel
with >= 30 variants we pretend that only a budget of k variants can be
synthesized and compare three ways of choosing them:

  model      non-dominated sorting on predicted (latency, resource) fronts
  heuristic  the same sorting on one pragma feature per objective, chosen and
             signed on the training kernels (as in the ranking heuristic)
  random     uniform random choice (mean over 50 draws)
  oracle  non-dominated sorting on the true values (upper bound)

Objectives: Worst-caseLatency and the ForgeHLS ResourceMetric, i.e. the mean
utilization of LUT, FF, DSP and BRAM_18K on the xcu280 (the released is_pareto
flag uses the same resource metric with best-case latency). The predicted
resource metric is computed from the four predicted resources.

Metrics per kernel, averaged over kernels and then over the five seeds:
  Recall  fraction of true Pareto designs that are among the k chosen
  ADRS    average distance from the true front to the best chosen design
          (Schafer-style: max relative excess in latency or resource, >= 0)

Output: hls_ml/results/pareto_dse_by_seed.csv, pareto_dse_summary.csv,
        pareto_dse_summary.txt
"""

import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES, SEEDS

# Diagnostic only: --keep-row-order skips the per-kernel shuffle to show how
# much the dataset's row order leaks into tie breaking. Outputs get a suffix.
KEEP_ROW_ORDER = "--keep-row-order" in sys.argv
SUFFIX = "_row_ordered" if KEEP_ROW_ORDER else ""
from train_baselines import fit_predict
from extra_analysis import choose_heuristic

AVAILABLE = {"LUT": 1303680, "FF": 2607360, "DSP": 9024, "BRAM_18K": 4032}
BUDGETS = [0.05, 0.10, 0.20]
MIN_ROWS = 30
RANDOM_DRAWS = 50
EPS = 1e-9


def resource_metric(lut, ff, dsp, bram):
    return (lut / AVAILABLE["LUT"] + ff / AVAILABLE["FF"]
            + dsp / AVAILABLE["DSP"] + bram / AVAILABLE["BRAM_18K"]) / 4.0


def pareto_mask(a, b):
    """Non-dominated points when minimizing both a and b."""
    order = np.lexsort((b, a))
    mask = np.zeros(len(a), dtype=bool)
    best_b = np.inf
    for i in order:
        if b[i] < best_b:
            mask[i] = True
            best_b = b[i]
    return mask


def front_order(a, b):
    """Indices ordered by non-dominated front; within a front by summed ranks.

    Ties keep the incoming order, so callers must shuffle rows first.
    """
    remaining = np.arange(len(a))
    out = []
    while len(remaining):
        m = pareto_mask(a[remaining], b[remaining])
        front = remaining[m]
        ra = np.argsort(np.argsort(a[front]))
        rb = np.argsort(np.argsort(b[front]))
        out.extend(front[np.argsort(ra + rb, kind="stable")])
        remaining = remaining[~m]
    return np.array(out)


def adrs(true_front, chosen):
    """true_front, chosen: arrays of shape (n, 2) with true objective values."""
    dist = []
    for p in true_front:
        rel = (chosen - p) / np.maximum(np.abs(p), EPS)
        dist.append(np.min(np.maximum(0.0, rel.max(axis=1))))
    return float(np.mean(dist))


def evaluate_kernel(lat, res, plat, pres, hlat, hres, rng):
    # ForgeHLS stores the Pareto variants of a kernel early in its row order.
    # Shuffle first so that ties in predicted scores are broken at random and
    # row order cannot leak Pareto membership into any method.
    if not KEEP_ROW_ORDER:
        perm = rng.permutation(len(lat))
        lat, res, plat, pres, hlat, hres = (a[perm] for a in (lat, res, plat, pres, hlat, hres))
    n = len(lat)
    true_mask = pareto_mask(lat, res)
    front = np.column_stack([lat[true_mask], res[true_mask]])
    truth = np.column_stack([lat, res])
    orders = {
        "Model": front_order(plat, pres),
        "Heuristic": front_order(hlat, hres),
        "Oracle": front_order(lat, res),
    }
    rows = []
    for budget in BUDGETS:
        k = max(1, int(np.ceil(budget * n)))
        rec = {"Budget": budget, "k": k, "N": n, "N_Pareto": int(true_mask.sum())}
        for name, order in orders.items():
            chosen = order[:k]
            rec[f"{name}_Recall"] = true_mask[chosen].sum() / true_mask.sum()
            rec[f"{name}_ADRS"] = adrs(front, truth[chosen])
        rr, ra = [], []
        for _ in range(RANDOM_DRAWS):
            chosen = rng.choice(n, size=k, replace=False)
            rr.append(true_mask[chosen].sum() / true_mask.sum())
            ra.append(adrs(front, truth[chosen]))
        rec["Random_Recall"] = float(np.mean(rr))
        rec["Random_ADRS"] = float(np.mean(ra))
        rows.append(rec)
    return rows


def main():
    data = np.load(DATA_DIR / "dataset_arrays.npz")
    splits = np.load(DATA_DIR / "splits.npz")
    X, Y, groups = data["X_features"], data["y_targets"], data["group_ids"]
    feature_names = data["feature_names"]
    t_idx = {name: i for i, name in enumerate(TARGET_NAMES)}
    rng = np.random.default_rng(2026)

    records = []
    heur_names = []
    for seed in SEEDS:
        tr = splits[f"e3_grouped_seed_{seed}_train_idx"]
        te = splits[f"e3_grouped_seed_{seed}_test_idx"]
        pred = {name: fit_predict("GradientBoosting", X, Y[:, i], tr, te, seed=seed)
                for name, i in t_idx.items()}
        y = {name: Y[te, i].astype(float) for name, i in t_idx.items()}
        true_res = resource_metric(y["LUT"], y["FF"], y["DSP"], y["BRAM_18K"])
        pred_res = resource_metric(pred["LUT"], pred["FF"], pred["DSP"], pred["BRAM_18K"])
        # Pragma heuristic: per objective, the single pragma feature whose
        # within-kernel ranking correlates best with that objective on the
        # TRAINING kernels, with its sign fixed there (lower score = better).
        tr_lat = Y[tr, t_idx["Worst-caseLatency"]].astype(float)
        tr_res = resource_metric(*(Y[tr, t_idx[c]].astype(float) for c in ("LUT", "FF", "DSP", "BRAM_18K")))
        heur = {}
        for obj, target in (("lat", tr_lat), ("res", tr_res)):
            fname, rho, col = choose_heuristic(X[tr], target, groups[tr], feature_names)
            heur[obj] = np.sign(rho) * X[te, col].astype(float)
            heur_names.append({"Seed": seed, "Objective": obj, "Feature": fname, "Train_rho": rho})
        g = groups[te]
        n_kernels = 0
        for kernel in np.unique(g):
            m = g == kernel
            if m.sum() < MIN_ROWS:
                continue
            n_kernels += 1
            for rec in evaluate_kernel(y["Worst-caseLatency"][m], true_res[m],
                                       pred["Worst-caseLatency"][m], pred_res[m],
                                       heur["lat"][m], heur["res"][m], rng):
                records.append({"Seed": seed, "Kernel": kernel, **rec})
        print(f"seed {seed}: {n_kernels} held-out kernels evaluated", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / f"pareto_dse_by_kernel{SUFFIX}.csv", index=False)
    metrics = [c for c in df.columns if c.endswith(("_Recall", "_ADRS"))]
    per_seed = df.groupby(["Seed", "Budget"])[metrics].mean()
    # ADRS means are dominated by a few kernels with huge relative distances,
    # so also keep the per-seed median over kernels.
    adrs_cols = [c for c in metrics if c.endswith("_ADRS")]
    medians = df.groupby(["Seed", "Budget"])[adrs_cols].median()
    medians.columns = [f"{c}Median" for c in adrs_cols]
    per_seed = per_seed.join(medians).reset_index()
    metrics = metrics + list(medians.columns)
    per_seed.to_csv(RESULTS_DIR / f"pareto_dse_by_seed{SUFFIX}.csv", index=False)
    summary = per_seed.groupby("Budget")[metrics].agg(["mean", lambda v: float(np.std(v, ddof=0))])
    summary.columns = [f"{a}_{'mean' if b == 'mean' else 'std'}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS_DIR / f"pareto_dse_summary{SUFFIX}.csv", index=False)

    pd.DataFrame(heur_names).to_csv(RESULTS_DIR / f"pareto_dse_heuristic_features{SUFFIX}.csv", index=False)
    # Share of kernels where the model finds more Pareto designs than random / heuristic.
    wins = df.assign(Win=df["Model_Recall"] > df["Random_Recall"]).groupby("Budget")["Win"].mean()
    wins_h = df.assign(Win=df["Model_Recall"] > df["Heuristic_Recall"]).groupby("Budget")["Win"].mean()
    ties_h = df.assign(T=df["Model_Recall"] == df["Heuristic_Recall"]).groupby("Budget")["T"].mean()

    lines = ["Budgeted DSE proxy on held-out kernels (E3, 5 seeds, GB)"
             + (" -- ROW ORDER KEPT (leaky diagnostic)" if KEEP_ROW_ORDER else ""),
             "=" * 75,
             f"Kernels per seed with >= {MIN_ROWS} variants; median true Pareto size "
             f"{df[df.Budget == BUDGETS[0]]['N_Pareto'].median():.0f}",
             "Heuristic features: " + ", ".join(
                 f"{o}:{f}" for o, f in pd.DataFrame(heur_names).groupby("Objective")["Feature"]
                 .agg(lambda v: "/".join(sorted(set(v)))).items()),
             f"{'Budget':>7} | {'Recall model':>13} {'heur.':>6} {'random':>7} {'oracle':>7} | "
             f"{'ADRS model':>13} {'heur.':>6} {'random':>7} {'oracle':>7} | model>rand  model>heur (ties)"]
    for _, r in summary.iterrows():
        b = r["Budget"]
        lines.append(
            f"{b:>7.0%} | {r['Model_Recall_mean']:>6.3f}±{r['Model_Recall_std']:.3f} "
            f"{r['Heuristic_Recall_mean']:>6.3f} {r['Random_Recall_mean']:>7.3f} {r['Oracle_Recall_mean']:>7.3f} | "
            f"{r['Model_ADRS_mean']:>6.3f}±{r['Model_ADRS_std']:.3f} "
            f"{r['Heuristic_ADRS_mean']:>6.3f} {r['Random_ADRS_mean']:>7.3f} {r['Oracle_ADRS_mean']:>7.3f} | "
            f"{wins.loc[b]:>9.1%}  {wins_h.loc[b]:>9.1%} ({ties_h.loc[b]:.1%})")
    lines.append("Median ADRS over kernels (mean over seeds): " + "; ".join(
        f"{r['Budget']:.0%}: model {r['Model_ADRSMedian_mean']:.3f}, heuristic {r['Heuristic_ADRSMedian_mean']:.3f}, "
        f"random {r['Random_ADRSMedian_mean']:.3f}, oracle {r['Oracle_ADRSMedian_mean']:.3f}"
        for _, r in summary.iterrows()))
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / f"pareto_dse_summary{SUFFIX}.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
