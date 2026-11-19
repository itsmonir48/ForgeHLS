"""
extra_analysis.py

Additional analyses reported in the paper that are not produced by the core
pipeline scripts. Every number here is recomputed from hls_ml/data and, for
the uncertainty part, from results/uncertainty_predictions.npz.

Sections
  [1] Dataset facts: constant features, feature-vector collisions and the
      error floor they imply, between-kernel share of target variance.
  [2] GB under five random-row splits (E1) and five kernel-disjoint splits
      (E3), with a constant training-median predictor as a floor.
  [3] Within-kernel ranking (median Spearman over test kernels with >= 30
      records) and a single-pragma-feature ranking heuristic.
  [4] Alternative group definitions: algorithm name only, and
      leave-one-source-out (LOSO).
  [5] Uncertainty vs. predicted magnitude on the E2 test set: magnitude
      baseline, within-decile correlation, kernel-resampling comparison,
      log-scale quintile error ratios, deep-ensemble log-sigma correlation.
  [6] Risk-coverage areas for tree log/raw sigma, predicted magnitude,
      oracle and deep log sigma (raw-MAE and log-MAE risk).

Run order: build_dataset.py -> make_splits.py -> train_uncertainty.py -> this.
Outputs go to hls_ml/results/extra_*.csv and extra_analysis_summary.txt.
"""

import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES, SEEDS
from train_baselines import fit_predict
from evaluate import accept_order

GB = "GradientBoosting"
MIN_KERNEL_ROWS = 30
N_RESAMPLES = 200
RC_COVERAGES = np.round(np.arange(0.05, 1.0001, 0.05), 2)
HEURISTIC_FEATURES = [
    "pragma_number",
    "pragma_pipeline_count",
    "pragma_unroll_count",
    "pragma_max_unroll_factor",
    "pragma_array_partition_count",
    "pragma_max_partition_factor",
]
SHORT = {"Worst-caseLatency": "Latency", "LUT": "LUT", "DSP": "DSP", "FF": "FF", "BRAM_18K": "BRAM"}

summary = []


def log(line=""):
    print(line, flush=True)
    summary.append(line)


def log_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None)))))


def constant_prediction(y_train, n):
    """Training median in log space, mapped back to the original scale."""
    return np.full(n, np.expm1(np.median(np.log1p(np.clip(y_train, 0, None)))))


def safe_spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def within_kernel_rho(y, score, groups, min_rows=MIN_KERNEL_ROWS):
    """Median Spearman(score, y) over kernels with >= min_rows rows."""
    rhos = []
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() < min_rows:
            continue
        r = safe_spearman(score[idx], y[idx])
        if not np.isnan(r):
            rhos.append(r)
    return (float(np.median(rhos)) if rhos else np.nan), len(rhos)


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def section_dataset(X, Y, groups, feature_names):
    log("[1] Dataset facts")
    const = [str(feature_names[i]) for i in range(X.shape[1]) if np.ptp(X[:, i]) == 0]
    log(f"  Constant features: {const if const else 'none'}")

    key = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).astype(str).to_numpy()
    key = np.char.add(np.char.add(key, "|"), groups.astype(str))
    counts = pd.Series(key).value_counts()
    colliding = int(counts[counts > 1].sum())
    log(f"  Distinct (feature vector, kernel) pairs: {len(counts):,} of {len(key):,} rows")
    log(f"  Rows sharing their feature vector with another row of the same kernel: "
        f"{colliding:,} ({colliding / len(key):.1%})")

    rows = []
    for t, name in enumerate(TARGET_NAMES):
        ly = pd.Series(np.log1p(Y[:, t]))
        floor = float(np.mean(np.abs(ly - ly.groupby(key).transform("median"))))
        g_mean = ly.groupby(groups).transform("mean")
        between = float(np.sum((g_mean - ly.mean()) ** 2) / np.sum((ly - ly.mean()) ** 2))
        zero = float(np.mean(Y[:, t] == 0))
        rows.append({"Target": name, "Collision_LogMAE_Floor": floor,
                     "Between_Kernel_Variance_Share": between, "Zero_Fraction": zero})
        log(f"  {SHORT[name]:<8} collision floor LogMAE={floor:.3f} | "
            f"between-kernel log-variance share={between:.1%} | zeros={zero:.1%}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "extra_dataset_facts.csv", index=False)
    log()


def choose_heuristic(Xtr, ytr, gtr, feature_names):
    """Pick the pragma feature with the largest |median within-kernel rho| on training kernels."""
    best = None
    for fname in HEURISTIC_FEATURES:
        col = list(feature_names).index(fname)
        rho, _ = within_kernel_rho(ytr, Xtr[:, col], gtr)
        if np.isnan(rho):
            continue
        if best is None or abs(rho) > abs(best[1]):
            best = (fname, rho, col)
    return best


def section_splits_and_ranking(X, Y, groups, feature_names):
    log("[2] GB under random-row (E1) and kernel-disjoint (E3) splits, five seeds each")
    log("[3] Within-kernel ranking and single-feature heuristic")
    splits = np.load(DATA_DIR / "splits.npz")
    idx_all = np.arange(len(Y))
    per_seed = []
    for seed in SEEDS:
        e1_tr, e1_te = train_test_split(idx_all, test_size=0.20, random_state=seed, shuffle=True)
        e3_tr = splits[f"e3_grouped_seed_{seed}_train_idx"]
        e3_te = splits[f"e3_grouped_seed_{seed}_test_idx"]
        for t, name in enumerate(TARGET_NAMES):
            y = Y[:, t]
            rec = {"Seed": seed, "Target": name}
            for tag, tr, te in (("E1", e1_tr, e1_te), ("E3", e3_tr, e3_te)):
                pred = fit_predict(GB, X, y, tr, te, seed=seed)
                const = constant_prediction(y[tr], len(te))
                rec[f"{tag}_LogMAE"] = log_mae(y[te], pred)
                rec[f"{tag}_R2"] = r2(y[te], pred)
                rec[f"{tag}_Const_LogMAE"] = log_mae(y[te], const)
                rho, n_k = within_kernel_rho(y[te], pred, groups[te])
                rec[f"{tag}_WithinKernel_rho"] = rho
                rec[f"{tag}_WithinKernel_n"] = n_k
            fname, train_rho, col = choose_heuristic(X[e3_tr], y[e3_tr], groups[e3_tr], feature_names)
            sign = 1.0 if train_rho >= 0 else -1.0
            h_rho, _ = within_kernel_rho(y[e3_te], sign * X[e3_te, col], groups[e3_te])
            rec["Heuristic_Feature"] = fname
            rec["Heuristic_Sign"] = sign
            rec["E3_Heuristic_rho"] = h_rho
            per_seed.append(rec)
        log(f"  seed {seed} done")
    df = pd.DataFrame(per_seed)
    df.to_csv(RESULTS_DIR / "extra_e1_e3_by_seed.csv", index=False)

    num = [c for c in df.columns if c not in ("Seed", "Target", "Heuristic_Feature")]
    agg = df.groupby("Target")[num].agg(["mean", lambda v: float(np.std(v, ddof=0))])
    agg.columns = [f"{a}_{'mean' if b == 'mean' else 'std'}" for a, b in agg.columns]
    agg["Heuristic_Feature"] = df.groupby("Target")["Heuristic_Feature"].agg(
        lambda v: ",".join(sorted(set(v))))
    agg = agg.reindex(TARGET_NAMES).reset_index()
    agg.to_csv(RESULTS_DIR / "extra_e1_e3_summary.csv", index=False)

    log(f"  {'Target':<8} {'E1 LogMAE':>14} {'E3 LogMAE':>14} {'Const E3':>9} {'ratio':>6} "
        f"{'E1 R2':>6} {'E3 R2':>7} {'rho E1/E3':>11} {'heur.':>6}  heuristic feature")
    for _, r in agg.iterrows():
        log(f"  {SHORT[r['Target']]:<8} {r['E1_LogMAE_mean']:.3f}±{r['E1_LogMAE_std']:.3f}   "
            f"{r['E3_LogMAE_mean']:.3f}±{r['E3_LogMAE_std']:.3f} {r['E3_Const_LogMAE_mean']:>9.3f} "
            f"{r['E3_LogMAE_mean'] / r['E1_LogMAE_mean']:>6.2f} {r['E1_R2_mean']:>6.2f} {r['E3_R2_mean']:>7.2f} "
            f"{r['E1_WithinKernel_rho_mean']:.2f}/{r['E3_WithinKernel_rho_mean']:.2f}"
            f"{r['E3_Heuristic_rho_mean']:>8.2f}  {r['Heuristic_Feature']}")
    for _, r in agg.iterrows():
        gain = 1 - r["E3_LogMAE_mean"] / r["E3_Const_LogMAE_mean"]
        log(f"  {SHORT[r['Target']]:<8} GB vs constant (E3): {gain:+.1%} LogMAE")
    log()
    return agg


def section_groupings(X, Y, groups, meta, e3_summary):
    log("[4] Alternative group definitions (GB LogMAE)")
    algo = meta["algo_name"].to_numpy()
    source = meta["source_name"].to_numpy()
    idx_all = np.arange(len(Y))

    algo_rows = []
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
                      .split(idx_all, groups=algo))
        assert not set(algo[tr]) & set(algo[te])
        for t, name in enumerate(TARGET_NAMES):
            pred = fit_predict(GB, X, Y[:, t], tr, te, seed=seed)
            algo_rows.append({"Seed": seed, "Target": name, "LogMAE": log_mae(Y[te, t], pred)})
    algo_df = pd.DataFrame(algo_rows)

    loso_rows = []
    for src in np.unique(source):
        te = idx_all[source == src]
        tr = idx_all[source != src]
        for t, name in enumerate(TARGET_NAMES):
            pred = fit_predict(GB, X, Y[:, t], tr, te, seed=42)
            loso_rows.append({
                "Source": src, "Target": name, "N_Test": len(te),
                "GB_LogMAE": log_mae(Y[te, t], pred),
                "Const_LogMAE": log_mae(Y[te, t], constant_prediction(Y[tr, t], len(te))),
            })
        log(f"  LOSO {src} done")
    loso_df = pd.DataFrame(loso_rows)
    loso_df.to_csv(RESULTS_DIR / "extra_loso_by_source.csv", index=False)

    rows = []
    for name in TARGET_NAMES:
        e3 = e3_summary.set_index("Target").loc[name]
        sub = loso_df[loso_df["Target"] == name]
        rows.append({
            "Target": name,
            "Constant_E3": e3["E3_Const_LogMAE_mean"],
            "SourceAlgo_E3": e3["E3_LogMAE_mean"],
            "AlgoName_5seed": algo_df[algo_df["Target"] == name]["LogMAE"].mean(),
            "LOSO_Median": sub["GB_LogMAE"].median(),
            "LOSO_Const_Median": sub["Const_LogMAE"].median(),
            "LOSO_Mean": sub["GB_LogMAE"].mean(),
            "LOSO_Const_Mean": sub["Const_LogMAE"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "extra_group_definitions.csv", index=False)
    log(f"  {'Target':<8} {'Const':>6} {'Src-algo':>9} {'Algo':>6} {'LOSO':>6} {'LOSO c.':>8}")
    for _, r in out.iterrows():
        log(f"  {SHORT[r['Target']]:<8} {r['Constant_E3']:>6.2f} {r['SourceAlgo_E3']:>9.2f} "
            f"{r['AlgoName_5seed']:>6.2f} {r['LOSO_Median']:>6.2f} {r['LOSO_Const_Median']:>8.2f}")
    worst = loso_df[loso_df["GB_LogMAE"] > loso_df["Const_LogMAE"]]
    for _, r in worst.iterrows():
        log(f"  LOSO worse than constant: {r['Source']} / {SHORT[r['Target']]}: "
            f"{r['GB_LogMAE']:.2f} vs {r['Const_LogMAE']:.2f}")
    log()


def aurc(risk_values, order, coverages=RC_COVERAGES):
    n = len(order)
    risks = [float(np.mean(risk_values[order[:max(1, int(np.ceil(c * n)))]])) for c in coverages]
    return float(np.trapezoid(risks, coverages)), risks


def section_uncertainty(groups):
    log("[5] Uncertainty vs predicted magnitude (E2 test set)")
    npz = np.load(RESULTS_DIR / "uncertainty_predictions.npz")
    splits = np.load(DATA_DIR / "splits.npz")
    test_groups = groups[splits["e2_grouped_test_idx"]]
    Y = npz["Y_test_true"]
    P = npz["rf_preds"]
    S = npz["rf_uncertainty"]
    S_log = npz["rf_uncertainty_log"]
    D = npz["deep_ensemble_preds"]
    D_log = npz["deep_ensemble_uncertainty_log"]
    D_raw = npz["deep_ensemble_uncertainty"]
    rng = np.random.default_rng(2026)
    uniq = np.unique(test_groups)
    members = {g: np.flatnonzero(test_groups == g) for g in uniq}

    unc_rows, quint_rows, aurc_rows, curve_rows = [], [], [], []
    for t, name in enumerate(TARGET_NAMES):
        y, p, s, sl = Y[:, t], P[:, t], S[:, t], S_log[:, t]
        err = np.abs(y - p)
        log_err = np.abs(np.log1p(y) - np.log1p(p))
        rho_tree = safe_spearman(s, err)
        rho_mag = safe_spearman(p, err)
        rho_s_mag = safe_spearman(s, p)
        rho_deep = safe_spearman(D_raw[:, t], np.abs(y - D[:, t]))
        rho_deep_log = safe_spearman(D_log[:, t], np.abs(np.log1p(y) - np.log1p(D[:, t])))
        rho_tree_log = safe_spearman(sl, log_err)

        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        dec_rhos = [safe_spearman(s[dec == d], err[dec == d]) for d in np.unique(dec)]
        dec_rhos = [r for r in dec_rhos if not np.isnan(r)]

        wins = 0
        for _ in range(N_RESAMPLES):
            pick = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
            if safe_spearman(s[pick], err[pick]) > safe_spearman(p[pick], err[pick]):
                wins += 1

        unc_rows.append({
            "Target": name, "Tree_sigma_rho": rho_tree, "Deep_sigma_rho": rho_deep,
            "Magnitude_rho": rho_mag, "Tree_sigma_vs_magnitude_rho": rho_s_mag,
            "Within_decile_tree_rho_mean": float(np.mean(dec_rhos)) if dec_rhos else np.nan,
            "Tree_beats_magnitude_frac": wins / N_RESAMPLES,
            "Tree_log_sigma_vs_log_err_rho": rho_tree_log,
            "Deep_log_sigma_vs_log_err_rho": rho_deep_log,
        })

        q = pd.qcut(s, 5, labels=False, duplicates="drop")
        q_means = [float(np.mean(log_err[q == k])) for k in np.unique(q)]
        for k, m in zip(np.unique(q), q_means):
            quint_rows.append({"Target": name, "Quintile_Bin": int(k), "Mean_Log_Error": m})
        ratio = q_means[-1] / q_means[0] if q_means[0] > 0 else np.inf

        log(f"  {SHORT[name]:<8} rho(tree σ,err)={rho_tree:.3f} rho(deep σ,err)={rho_deep:.3f} "
            f"rho(|ŷ|,err)={rho_mag:.3f} rho(σ,|ŷ|)={rho_s_mag:.3f} | within-decile={unc_rows[-1]['Within_decile_tree_rho_mean']:.2f} "
            f"| tree beats |ŷ| in {wins / N_RESAMPLES:.1%} | deep log ρ={rho_deep_log:+.2f} "
            f"| Q5/Q1 log err={ratio:.1f} ({len(q_means)} bins)")

        # Random tie breaking: row order in ForgeHLS is not random (Pareto
        # variants come first within a kernel) and many scores are tied.
        orders = {
            "Tree_log_sigma": accept_order(sl),
            "Tree_raw_sigma": accept_order(s),
            "Magnitude": accept_order(p),
            "Oracle": accept_order(err),
            "Deep_log_sigma": accept_order(D_log[:, t]),
        }
        deep_err = np.abs(y - D[:, t])
        rec_raw, rec_log = {"Target": name, "Risk": "raw_MAE"}, {"Target": name, "Risk": "log_MAE"}
        span = RC_COVERAGES[-1] - RC_COVERAGES[0]
        for method, order in orders.items():
            e_raw = deep_err if method == "Deep_log_sigma" else err
            e_log = (np.abs(np.log1p(y) - np.log1p(D[:, t])) if method == "Deep_log_sigma" else log_err)
            a_raw, risks_raw = aurc(e_raw, order)
            a_log, _ = aurc(e_log, order)
            # Random rejection keeps the full-coverage mean risk in expectation.
            rec_raw[method] = (np.mean(e_raw) * span) / a_raw if a_raw > 0 else np.inf
            rec_log[method] = (np.mean(e_log) * span) / a_log if a_log > 0 else np.inf
            full = np.mean(e_raw)
            for c, r in zip(RC_COVERAGES, risks_raw):
                curve_rows.append({"Target": name, "Method": method, "Coverage": c,
                                   "Retained_MAE_Ratio": r / full if full > 0 else np.nan})
        aurc_rows += [rec_raw, rec_log]

    pd.DataFrame(unc_rows).to_csv(RESULTS_DIR / "extra_uncertainty_magnitude.csv", index=False)
    pd.DataFrame(quint_rows).to_csv(RESULTS_DIR / "extra_uncertainty_quintiles.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(RESULTS_DIR / "extra_risk_coverage_curves.csv", index=False)
    aurc_df = pd.DataFrame(aurc_rows)
    aurc_df.to_csv(RESULTS_DIR / "extra_risk_coverage_aurc.csv", index=False)
    log()
    log("[6] AURC reduction vs random rejection (×, coverage 5%..100%)")
    for _, r in aurc_df.iterrows():
        log(f"  {SHORT[r['Target']]:<8} {r['Risk']:<8} tree log σ={r['Tree_log_sigma']:.2f} "
            f"tree raw σ={r['Tree_raw_sigma']:.2f} |ŷ|={r['Magnitude']:.2f} "
            f"oracle={r['Oracle']:.2f} deep log σ={r['Deep_log_sigma']:.2f}")
    at80 = curves[(curves["Method"] == "Deep_log_sigma") & np.isclose(curves["Coverage"], 0.8)]
    for _, r in at80.iterrows():
        log(f"  Deep log σ, {SHORT[r['Target']]}: retained MAE at 80% coverage = "
            f"{r['Retained_MAE_Ratio']:.2f}× full-coverage MAE")
    log()


def main():
    t0 = time.time()
    log("Extra analyses for the paper")
    log("=" * 75)
    data = np.load(DATA_DIR / "dataset_arrays.npz", allow_pickle=False)
    X, Y = data["X_features"], data["y_targets"]
    groups = data["group_ids"]
    feature_names = data["feature_names"]
    meta = pq.read_table(DATA_DIR / "dataset_metadata.parquet").to_pandas()
    assert (meta["group_id"].to_numpy() == groups).all()

    section_dataset(X, Y, groups, feature_names)
    e3_summary = section_splits_and_ranking(X, Y, groups, feature_names)
    section_groupings(X, Y, groups, meta, e3_summary)
    section_uncertainty(groups)

    log(f"Finished in {time.time() - t0:.0f}s")
    (RESULTS_DIR / "extra_analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
