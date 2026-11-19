"""
deep_ensemble_stability.py

Measures how much the deep-ensemble results depend on the random seeds of its
members. The 5-member MLP ensemble of train_uncertainty.py is retrained with
three disjoint seed sets on the same E2 fit split, and for each run we report
  - Spearman(raw sigma, |error|) and Spearman(log sigma, |log error|)
  - AURC reduction vs. random rejection when accepting the lowest log sigma first
    (raw-MAE risk, coverage 5%..100%)
  - retained MAE at 80% coverage relative to full coverage
  - ENCE on the log scale

Output: hls_ml/results/deep_ensemble_stability.csv
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES
from evaluate import compute_ence, accept_order
from train_uncertainty import train_single_nn, DEVICE

SEED_SETS = {
    "base100": [100, 107, 114, 121, 128],  # the seeds used in train_uncertainty.py
    "base200": [200, 207, 214, 221, 228],
    "base300": [300, 307, 314, 321, 328],
}
COVERAGES = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def aurc_reduction(err, score):
    order = accept_order(score)
    n = len(order)
    risks = [np.mean(err[order[:max(1, int(np.ceil(c * n)))]]) for c in COVERAGES]
    area = np.trapezoid(risks, COVERAGES)
    k80 = int(np.ceil(0.8 * n))
    return (np.mean(err) * (COVERAGES[-1] - COVERAGES[0])) / area, np.mean(err[order[:k80]]) / np.mean(err)


def main():
    data = np.load(DATA_DIR / "dataset_arrays.npz")
    splits = np.load(DATA_DIR / "splits.npz")
    X, Y = data["X_features"], data["y_targets"]
    tr, te = splits["e2_grouped_train_idx"], splits["e2_grouped_test_idx"]
    fit_local, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2026)
                        .split(X[tr], Y[tr], groups=data["group_ids"][tr]))
    X_fit, Y_fit, X_test, Y_test = X[tr][fit_local], Y[tr][fit_local], X[te], Y[te]

    xs, ys = StandardScaler(), StandardScaler()
    Xf = xs.fit_transform(X_fit)
    Xt = torch.tensor(xs.transform(X_test), dtype=torch.float32, device=DEVICE)
    Yf = ys.fit_transform(np.log1p(np.clip(Y_fit, 0, None)))
    max_log = np.log1p(np.max(np.clip(Y_fit, 0, None), axis=0))

    rows = []
    for name, seeds in SEED_SETS.items():
        preds = []
        for seed in seeds:
            model = train_single_nn(Xf, Yf, in_dim=X.shape[1], out_dim=len(TARGET_NAMES), seed=seed)
            with torch.no_grad():
                p = ys.inverse_transform(model(Xt).cpu().numpy())
            preds.append(np.clip(np.expm1(np.clip(p, 0.0, max_log)), 0, None))
        preds = np.array(preds)
        mean, sd, log_sd = preds.mean(0), preds.std(0), np.log1p(preds).std(0)
        for t, target in enumerate(TARGET_NAMES):
            y = Y_test[:, t]
            err = np.abs(y - mean[:, t])
            log_err = np.abs(np.log1p(y) - np.log1p(mean[:, t]))
            red, at80 = aurc_reduction(err, log_sd[:, t])
            rows.append({
                "Seed_Set": name, "Target": target,
                "LogMAE": float(np.mean(log_err)),
                "rho_raw_sigma_err": float(spearmanr(sd[:, t], err)[0]),
                "rho_log_sigma_log_err": float(spearmanr(log_sd[:, t], log_err)[0]),
                "AURC_reduction_log_sigma": float(red),
                "Retained_MAE_at_80": float(at80),
                "ENCE_log": compute_ence(np.log1p(y), np.log1p(mean[:, t]), log_sd[:, t]),
            })
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "deep_ensemble_stability.csv", index=False)
    cols = ["LogMAE", "rho_raw_sigma_err", "rho_log_sigma_log_err",
            "AURC_reduction_log_sigma", "Retained_MAE_at_80", "ENCE_log"]
    print(df.groupby("Target")[cols].agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
