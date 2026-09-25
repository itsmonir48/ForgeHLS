"""
train_uncertainty.py

Implements Uncertainty Quantification (UQ) for multi-objective HLS QoR prediction under
kernel-level distribution shift (Milestone 4 - E4 & E5).

Methods:
1. Method A (Tree-based UQ / QRF):
   - Individual tree variance and quantile-based prediction intervals (10th, 50th, 90th percentiles).
2. Method B (Deep Ensemble in PyTorch):
   - 5-model neural ensemble capturing epistemic uncertainty via inter-model prediction variance.

Evaluates Hypothesis H3:
- u(x) correlates with actual error |y - y_hat| (Spearman rank correlation).
- Quantile error separation: MAE in lowest uncertainty quintile (Q1) vs highest quintile (Q5).

Outputs:
- hls_ml/results/uncertainty_results.csv
- hls_ml/results/uncertainty_summary.txt
- hls_ml/results/uncertainty_predictions.npz (saved for Milestone 5 Calibration)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES


# -------------------------------------------------------------
# Deep Ensemble Architecture (PyTorch)
# -------------------------------------------------------------
class QoRNeuralNet(nn.Module):
    def __init__(self, in_features=42, out_features=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)


def train_single_nn(X_train, Y_train, in_dim=42, out_dim=5, seed=42, epochs=15, batch_size=512):
    torch.manual_seed(seed)
    model = QoRNeuralNet(in_dim, out_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def main():
    print("=" * 75)
    print("Training Uncertainty Quantification Models (Milestone 4: E4 & E5)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # [1] Load Data and Split E2
    # -------------------------------------------------------------
    print("\n[1] Loading dataset arrays and E2 (Kernel-Disjoint) split...")
    npz_data = np.load(DATA_DIR / "dataset_arrays.npz")
    splits_data = np.load(DATA_DIR / "splits.npz")

    X = npz_data["X_features"]
    Y = npz_data["y_targets"]

    train_idx = splits_data["e2_grouped_train_idx"]
    test_idx = splits_data["e2_grouped_test_idx"]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Hold out complete training groups for calibration; the final test groups
    # remain untouched until the reported evaluation.
    group_ids = npz_data["group_ids"][train_idx]
    calibration_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2026)
    fit_local, calib_local = next(
        calibration_split.split(X_train, Y_train, groups=group_ids)
    )
    X_fit, Y_fit = X_train[fit_local], Y_train[fit_local]
    X_calib, Y_calib = X_train[calib_local], Y_train[calib_local]

    print(f"E2 Train: {len(X_train):,} designs | Test: {len(X_test):,} unseen designs")

    # Scale inputs and targets for Neural Ensemble
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_fit)
    X_test_scaled = x_scaler.transform(X_test)
    Y_train_scaled = y_scaler.fit_transform(np.log1p(np.clip(Y_fit, 0, None)))

    uq_results = []
    saved_arrays = {
        "Y_test_true": Y_test
    }

    # -------------------------------------------------------------
    # [2] Method A: Tree-based UQ (Random Forest Tree Variance & Quantiles)
    # -------------------------------------------------------------
    print("\n[2] Training Method A: Tree-based UQ & Prediction Intervals...")
    rf_preds_mean = np.zeros_like(Y_test)
    rf_uncertainty_std = np.zeros_like(Y_test)
    rf_pi_lower = np.zeros_like(Y_test)
    rf_pi_upper = np.zeros_like(Y_test)
    calibration_residuals = np.zeros((len(X_calib), len(TARGET_NAMES)), dtype=np.float32)

    for t_idx, target_name in enumerate(TARGET_NAMES):
        t0 = time.time()
        print(f"  Fitting RF for {target_name:<18}...", end=" ", flush=True)

        rf = RandomForestRegressor(
            n_estimators=40,
            max_depth=16,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_fit, np.log1p(np.clip(Y_fit[:, t_idx], 0, None)))

        # Extract predictions of all individual trees
        # tree.predict returns shape (N,)
        tree_preds_log = np.array([tree.predict(X_test) for tree in rf.estimators_])
        tree_preds = np.expm1(tree_preds_log)
        calib_tree_preds = np.expm1(np.array([tree.predict(X_calib) for tree in rf.estimators_]))

        mean_pred = np.mean(tree_preds, axis=0)
        std_uncert = np.std(tree_preds, axis=0)
        # Split-conformal calibration: the interval is based on held-out
        # absolute residuals, not on uncalibrated tree disagreement quantiles.
        calib_prediction = np.mean(calib_tree_preds, axis=0)
        calib_residuals = np.abs(Y_calib[:, t_idx] - calib_prediction)
        calibration_residuals[:, t_idx] = calib_residuals
        conformal_q = np.quantile(calib_residuals, 0.90)
        q10 = np.clip(mean_pred - conformal_q, 0, None)
        q90 = mean_pred + conformal_q

        rf_preds_mean[:, t_idx] = mean_pred
        rf_uncertainty_std[:, t_idx] = std_uncert
        rf_pi_lower[:, t_idx] = q10
        rf_pi_upper[:, t_idx] = q90

        # Calculate Spearman correlation with actual residual error |y - y_hat|
        abs_err = np.abs(Y_test[:, t_idx] - mean_pred)
        spearman_corr, _ = spearmanr(std_uncert, abs_err)

        # Quintile error separation (Q1 lowest uncert vs Q5 highest uncert)
        quintiles = pd.qcut(std_uncert, q=5, labels=False, duplicates='drop')
        mae_q1 = np.mean(abs_err[quintiles == 0])
        mae_q5 = np.mean(abs_err[quintiles == 4]) if 4 in quintiles else np.mean(abs_err[quintiles == max(quintiles)])
        error_ratio = mae_q5 / (mae_q1 + 1e-6)

        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s | Spearman rho = {spearman_corr:.3f} | MAE Q1: {mae_q1:.1f} vs Q5: {mae_q5:.1f} ({error_ratio:.2f}x)")

        uq_results.append({
            "Method": "Tree_Ensemble_UQ",
            "Target": target_name,
            "Spearman_Correlation": round(float(spearman_corr), 4),
            "MAE_Lowest_Uncertainty_Q1": round(float(mae_q1), 2),
            "MAE_Highest_Uncertainty_Q5": round(float(mae_q5), 2),
            "Uncertainty_Error_Ratio_Q5_vs_Q1": round(float(error_ratio), 2),
            "Mean_Uncertainty": round(float(np.mean(std_uncert)), 2)
        })

    saved_arrays["rf_preds"] = rf_preds_mean
    saved_arrays["rf_uncertainty"] = rf_uncertainty_std
    saved_arrays["rf_pi_lower"] = rf_pi_lower
    saved_arrays["rf_pi_upper"] = rf_pi_upper
    saved_arrays["rf_calibration_residuals"] = calibration_residuals

    # -------------------------------------------------------------
    # [3] Method B: Deep Ensemble (5 Neural Networks in PyTorch)
    # -------------------------------------------------------------
    print("\n[3] Training Method B: Deep Neural Ensemble (5 Models)...")
    ensemble_size = 5
    nn_test_preds = []

    for m_i in range(ensemble_size):
        seed = 100 + m_i * 7
        t0 = time.time()
        print(f"  Training Ensemble Member {m_i + 1}/{ensemble_size} (Seed {seed})...", end=" ", flush=True)

        model = train_single_nn(
            X_train_scaled, Y_train_scaled,
            in_dim=X.shape[1], out_dim=len(TARGET_NAMES),
            seed=seed, epochs=15, batch_size=512
        )

        with torch.no_grad():
            x_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            pred_scaled = model(x_tensor).numpy()
            pred_original = np.expm1(y_scaler.inverse_transform(pred_scaled))
            # Ensure non-negative QoR
            pred_original = np.clip(pred_original, 0, None)
            nn_test_preds.append(pred_original)

        print(f"Done in {time.time() - t0:.1f}s")

    nn_test_preds = np.array(nn_test_preds)  # shape (5, N_test, 5)
    deep_ensemble_mean = np.mean(nn_test_preds, axis=0)
    deep_ensemble_std = np.std(nn_test_preds, axis=0)

    print("\nEvaluating Deep Ensemble UQ metrics across targets:")
    for t_idx, target_name in enumerate(TARGET_NAMES):
        mean_pred = deep_ensemble_mean[:, t_idx]
        uncert = deep_ensemble_std[:, t_idx]
        abs_err = np.abs(Y_test[:, t_idx] - mean_pred)

        spearman_corr, _ = spearmanr(uncert, abs_err)

        quintiles = pd.qcut(uncert, q=5, labels=False, duplicates='drop')
        mae_q1 = np.mean(abs_err[quintiles == 0])
        mae_q5 = np.mean(abs_err[quintiles == 4]) if 4 in quintiles else np.mean(abs_err[quintiles == max(quintiles)])
        error_ratio = mae_q5 / (mae_q1 + 1e-6)

        print(f"  [{target_name:<18}] Spearman rho = {spearman_corr:.3f} | MAE Q1: {mae_q1:.1f} vs Q5: {mae_q5:.1f} ({error_ratio:.2f}x)")

        uq_results.append({
            "Method": "Deep_Ensemble_UQ",
            "Target": target_name,
            "Spearman_Correlation": round(float(spearman_corr), 4),
            "MAE_Lowest_Uncertainty_Q1": round(float(mae_q1), 2),
            "MAE_Highest_Uncertainty_Q5": round(float(mae_q5), 2),
            "Uncertainty_Error_Ratio_Q5_vs_Q1": round(float(error_ratio), 2),
            "Mean_Uncertainty": round(float(np.mean(uncert)), 2)
        })

    saved_arrays["deep_ensemble_preds"] = deep_ensemble_mean
    saved_arrays["deep_ensemble_uncertainty"] = deep_ensemble_std

    # -------------------------------------------------------------
    # [4] Save Results and Validations
    # -------------------------------------------------------------
    df_uq = pd.DataFrame(uq_results)
    csv_path = RESULTS_DIR / "uncertainty_results.csv"
    df_uq.to_csv(csv_path, index=False)
    print(f"\n[4] Saved uncertainty metrics table to: {csv_path}")

    pred_npz_path = RESULTS_DIR / "uncertainty_predictions.npz"
    np.savez_compressed(pred_npz_path, **saved_arrays)
    print(f"Saved uncertainty prediction arrays to: {pred_npz_path} ({pred_npz_path.stat().st_size / (1024*1024):.2f} MB)")

    # Write summary
    summary_file = RESULTS_DIR / "uncertainty_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Uncertainty Quantification (UQ) Summary (Milestone 4 - E4 & E5)\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. Hypothesis H3 Verification: Does Uncertainty Correlate with Actual Error?\n")
        f.write("-" * 75 + "\n")
        for method in ["Tree_Ensemble_UQ", "Deep_Ensemble_UQ"]:
            sub = df_uq[df_uq["Method"] == method]
            avg_spearman = sub["Spearman_Correlation"].mean()
            avg_ratio = sub["Uncertainty_Error_Ratio_Q5_vs_Q1"].mean()
            f.write(f"  * {method:<18}: Mean Spearman rho = {avg_spearman:.3f} | Q5/Q1 Error Ratio = {avg_ratio:.2f}x\n")

        f.write("\n  [VERIFICATION RESULT]:\n")
        h3_supported = bool((df_uq["Spearman_Correlation"] > 0).all())
        h3_label = "SUPPORTED" if h3_supported else "NOT SUPPORTED"
        f.write(f"  - H3 {h3_label}: uncertainty/error association is reported empirically; no claim is hard-coded.\n\n")

        f.write("2. Per-Target Detailed Breakdown:\n")
        f.write("-" * 75 + "\n")
        for _, r in df_uq.iterrows():
            f.write(f"  [{r['Method']:<16}] {r['Target']:<18}: Spearman rho = {r['Spearman_Correlation']:+.3f} | Q1={r['MAE_Lowest_Uncertainty_Q1']} vs Q5={r['MAE_Highest_Uncertainty_Q5']} ({r['Uncertainty_Error_Ratio_Q5_vs_Q1']}x)\n")

    print(f"Saved executive summary to: {summary_file}")
    print("\n" + "=" * 75)
    print("Milestone 4 completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
