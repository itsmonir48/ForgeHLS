#!/usr/bin/env python3
"""
extra_analysis.py -- additional checks reported in the paper (Sections V-A, V-C, V-D, V-E).

Run from anywhere:   python hls_ml/scripts/extra_analysis.py            (full, ~10 min on 1 CPU)
                     python hls_ml/scripts/extra_analysis.py --quick     (smoke test, ~2 min)
                     python hls_ml/scripts/extra_analysis.py --skip-training   (saved-prediction analyses only)

Requires data/dataset_arrays.npz, data/splits.npz and results/uncertainty_predictions.npz
(produced by the main pipeline). Writes CSVs to results/extra/.

Part 1  (saved predictions, no training)
  - variance decomposition (between- vs within-kernel share of log-target variance)
  - uncertainty vs. predicted-magnitude baseline (Spearman, partial Spearman, log-scale)
  - log-scale mean-error ratio between top and bottom uncertainty quintiles
  - area under the risk-coverage curve (AURC) for: tree log-sigma, tree raw-sigma, |y_hat|, oracle, deep log-sigma
  - group-bootstrap CI: rho(tree sigma, error) vs rho(|y_hat|, error)
Part 2  (trains HistGradientBoosting with the paper's GB configuration)
  A  E1 random row split, seeds 42-46
  B  E3 kernel-disjoint split (source::algo), seeds 42-46
  C  grouping by algorithm name only, seeds 42-46
  D  leave-one-source-out (11 folds)
  Each reports LogMAE, raw R2, constant (train-median) LogMAE and within-kernel median Spearman.
Part 3  within-kernel ranking against a trivial pragma heuristic (E3 seeds): for each target the single pragma
        feature and sign with the best median within-kernel Spearman on the TRAIN kernels is applied to the test kernels.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]          # .../hls_ml
DATA, RES = ROOT / "data", ROOT / "results"
OUT = RES / "extra"
OUT.mkdir(exist_ok=True)

TARGETS = ["Latency", "LUT", "DSP", "FF", "BRAM"]
SEEDS = [42, 43, 44, 45, 46]
COVS = np.linspace(0.1, 1.0, 10)

arr = np.load(DATA / "dataset_arrays.npz")
spl = np.load(DATA / "splits.npz")
X, Y, G = arr["X_features"], arr["y_targets"], arr["group_ids"]
LY = np.log1p(np.clip(Y, 0, None))
SRC = np.array([g.split("::")[0] for g in G])
ALGO = np.array([g.split("::", 1)[1] for g in G])


# --------------------------------------------------------------------------- helpers
def rk(x):
    return rankdata(x) / len(x)


def partial_spearman(u, e, m):
    ru, re_, rm = rk(u), rk(e), rk(m)
    res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
    return np.corrcoef(res(ru, rm), res(re_, rm))[0, 1]


def aurc(err, score):
    order = np.argsort(score, kind="stable")
    n = len(err)
    risk = [err[order[: max(1, int(np.ceil(c * n)))]].mean() for c in COVS]
    return np.trapezoid(risk, COVS)


def within_kernel_rho(y_true, y_pred, grp, min_n=30):
    out = []
    for k in np.unique(grp):
        ii = np.where(grp == k)[0]
        if len(ii) >= min_n and np.std(y_true[ii]) > 0 and np.std(y_pred[ii]) > 0:
            r = spearmanr(y_true[ii], y_pred[ii])[0]
            if np.isfinite(r):
                out.append(r)
    return (float(np.median(out)) if out else np.nan), len(out)


# --------------------------------------------------------------------------- part 1
def part1():
    # variance decomposition (log scale)
    df = pd.DataFrame(LY, columns=TARGETS)
    df["g"] = G
    rows = []
    for t in TARGETS:
        within = df.groupby("g")[t].var().mul(df.groupby("g").size()).sum() / len(df)
        rows.append(dict(target=t, frac_between_kernels=1 - within / df[t].var(), frac_within=within / df[t].var()))
    pd.DataFrame(rows).round(3).to_csv(OUT / "variance_decomposition.csv", index=False)

    d = np.load(RES / "uncertainty_predictions.npz")
    te = spl["e2_grouped_test_idx"]
    g_te, Yt = G[te], d["Y_test_true"]
    assert np.allclose(Yt, Y[te])
    groups = np.unique(g_te)
    idx_by = {k: np.where(g_te == k)[0] for k in groups}
    rng = np.random.default_rng(0)

    corr, au, boot, quint = [], [], [], []
    for meth, p, u, ul in [("Tree", d["rf_preds"], d["rf_uncertainty"], d["rf_uncertainty_log"]),
                           ("Deep", d["deep_ensemble_preds"], d["deep_ensemble_uncertainty"],
                            d["deep_ensemble_uncertainty_log"])]:
        for j, t in enumerate(TARGETS):
            err = np.abs(Yt[:, j] - p[:, j])
            lerr = np.abs(np.log1p(Yt[:, j]) - np.log1p(p[:, j]))
            corr.append(dict(method=meth, target=t,
                             rho_raw_sigma=spearmanr(u[:, j], err)[0],
                             rho_log_sigma=spearmanr(ul[:, j], err)[0],
                             rho_magnitude=spearmanr(p[:, j], err)[0],
                             rho_sigma_vs_magnitude=spearmanr(u[:, j], p[:, j])[0],
                             partial_raw_sigma_given_magnitude=partial_spearman(u[:, j], err, p[:, j]),
                             rho_loglog=spearmanr(ul[:, j], np.log1p(err))[0]))
            rnd = np.trapezoid([err.mean()] * len(COVS), COVS)
            rnd_l = np.trapezoid([lerr.mean()] * len(COVS), COVS)
            au.append(dict(method=meth, target=t,
                           rand_over_log_sigma=rnd / aurc(err, ul[:, j]),
                           rand_over_raw_sigma=rnd / aurc(err, u[:, j]),
                           rand_over_magnitude=rnd / aurc(err, p[:, j]),
                           rand_over_oracle=rnd / aurc(err, err),
                           logrisk_rand_over_log_sigma=rnd_l / aurc(lerr, ul[:, j]),
                           logrisk_rand_over_magnitude=rnd_l / aurc(lerr, p[:, j]),
                           logrisk_rand_over_oracle=rnd_l / aurc(lerr, lerr)))
            if meth == "Tree":
                q = pd.qcut(u[:, j], 5, labels=False, duplicates="drop")
                quint.append(dict(target=t, n_bins=int(q.max()) + 1,
                                  logerr_Q1=lerr[q == 0].mean(), logerr_Q5=lerr[q == q.max()].mean(),
                                  ratio_Q5_over_Q1=lerr[q == q.max()].mean() / lerr[q == 0].mean()))
                est, mag = [], []
                for _ in range(200):
                    sel = rng.choice(groups, len(groups), replace=True)
                    ii = np.concatenate([idx_by[k] for k in sel])
                    ii = rng.choice(ii, min(len(ii), 20000), replace=False)
                    est.append(spearmanr(u[ii, j], err[ii])[0])
                    mag.append(spearmanr(p[ii, j], err[ii])[0])
                est, mag = np.array(est), np.array(mag)
                boot.append(dict(target=t, rho_sigma=spearmanr(u[:, j], err)[0],
                                 ci_sigma=f"[{np.percentile(est, 2.5):.3f},{np.percentile(est, 97.5):.3f}]",
                                 rho_magnitude=spearmanr(p[:, j], err)[0],
                                 ci_magnitude=f"[{np.percentile(mag, 2.5):.3f},{np.percentile(mag, 97.5):.3f}]",
                                 P_sigma_gt_magnitude=float(np.mean(est > mag))))
    for name, rows in [("uncertainty_vs_magnitude", corr), ("aurc_vs_magnitude", au),
                       ("quintile_log_error_ratio", quint), ("group_bootstrap_rho", boot)]:
        pd.DataFrame(rows).round(3).to_csv(OUT / f"{name}.csv", index=False)
    print("part 1 done ->", OUT)


# --------------------------------------------------------------------------- part 2
def fit_eval(tr, te, j, seed):
    m = HistGradientBoostingRegressor(max_iter=100, max_depth=12, learning_rate=0.1, random_state=seed)
    m.fit(X[tr], LY[tr, j])
    lp = np.clip(m.predict(X[te]), 0, LY[tr, j].max())
    wk, ng = within_kernel_rho(LY[te, j], lp, G[te])
    return dict(LogMAE=np.mean(np.abs(LY[te, j] - lp)),
                const_LogMAE=np.mean(np.abs(LY[te, j] - np.median(LY[tr, j]))),
                R2_raw=r2_score(Y[te, j], np.expm1(lp)), within_kernel_rho=wk, n_kernels_rho=ng)


def part2(quick):
    seeds = SEEDS[:1] if quick else SEEDS
    rows = []
    for seed in seeds:                                              # A: random rows
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
        for j, t in enumerate(TARGETS):
            rows.append(dict(exp="A_E1_random_rows", split=seed, target=t, **fit_eval(tr, te, j, seed)))
    for seed in seeds:                                              # B: paper's E3
        tr, te = spl[f"e3_grouped_seed_{seed}_train_idx"], spl[f"e3_grouped_seed_{seed}_test_idx"]
        for j, t in enumerate(TARGETS):
            rows.append(dict(exp="B_E3_source_algo", split=seed, target=t, **fit_eval(tr, te, j, seed)))
    for seed in seeds:                                              # C: algorithm-name groups
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=ALGO))
        assert not set(ALGO[tr]) & set(ALGO[te])
        for j, t in enumerate(TARGETS):
            rows.append(dict(exp="C_algo_name_only", split=seed, target=t, **fit_eval(tr, te, j, seed)))
    sources = np.unique(SRC)[:2] if quick else np.unique(SRC)       # D: leave-one-source-out
    for s in sources:
        te, tr = np.where(SRC == s)[0], np.where(SRC != s)[0]
        for j, t in enumerate(TARGETS):
            rows.append(dict(exp="D_leave_one_source_out", split=s, target=t, n_test_rows=len(te),
                             **fit_eval(tr, te, j, 42)))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "robustness_raw.csv", index=False)
    summ = raw.groupby(["exp", "target"]).agg(
        LogMAE=("LogMAE", "mean"), LogMAE_sd=("LogMAE", "std"), LogMAE_median=("LogMAE", "median"),
        const_LogMAE=("const_LogMAE", "mean"), const_LogMAE_median=("const_LogMAE", "median"),
        R2=("R2_raw", "mean"), R2_sd=("R2_raw", "std"), within_kernel_rho=("within_kernel_rho", "mean")).round(3)
    summ.to_csv(OUT / "robustness_summary.csv")
    print(summ.to_string())
    print("part 2 done ->", OUT)


# --------------------------------------------------------------------------- part 3
HEURISTIC_FEATURES = ["pragma_number", "pragma_max_unroll_factor", "pragma_max_partition_factor",
                      "pragma_pipeline_count", "pragma_unroll_count", "pragma_array_partition_count"]


def part3():
    names = [str(n) for n in arr["feature_names"]]
    cols = [names.index(f) for f in HEURISTIC_FEATURES]
    rows = []
    for seed in SEEDS:
        tr, te = spl[f"e3_grouped_seed_{seed}_train_idx"], spl[f"e3_grouped_seed_{seed}_test_idx"]
        for j, t in enumerate(TARGETS):
            best = (0.0, None, 0)
            for f, c in zip(HEURISTIC_FEATURES, cols):
                rho_tr, n_tr = within_kernel_rho(LY[tr, j], X[tr, c], G[tr])
                if np.isfinite(rho_tr) and abs(rho_tr) > abs(best[0]):
                    best = (rho_tr, f, c)
            if best[1] is None:
                rows.append(dict(split=seed, target=t, feature=None, sign=0, heuristic_rho=np.nan, n_kernels=0))
                continue
            sign = 1.0 if best[0] > 0 else -1.0
            rho_te, n_te = within_kernel_rho(LY[te, j], sign * X[te, best[2]], G[te])
            rows.append(dict(split=seed, target=t, feature=best[1], sign=sign, heuristic_rho=rho_te, n_kernels=n_te))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "within_kernel_heuristic_raw.csv", index=False)
    summ = raw.groupby("target").agg(heuristic_rho=("heuristic_rho", "mean"), n_kernels=("n_kernels", "mean"),
                                      feature=("feature", lambda s: s.mode().iloc[0] if s.notna().any() else None)).round(3)
    summ.to_csv(OUT / "within_kernel_heuristic_summary.csv")
    print(summ.to_string())
    print("part 3 done ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: one seed, two LOSO folds")
    ap.add_argument("--skip-training", action="store_true", help="only run saved-prediction analyses")
    a = ap.parse_args()
    part1()
    part3()
    if not a.skip_training:
        part2(a.quick)
