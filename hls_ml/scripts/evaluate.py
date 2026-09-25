"""
evaluate.py

Calibration & Risk-Coverage Selective Prediction Framework (Milestone 5: E6 & E7).

Evaluates:
- E6: Calibration Metrics
  * PICP (Prediction Interval Coverage Probability)
  * MPIW (Mean Prediction Interval Width)
  * ENCE (Expected Normalized Calibration Error)
  * Coverage vs Nominal Probability across percentiles (50%, 60%, 70%, 80%, 90%, 95%)

- E7: Risk-Coverage Selective Prediction (Abstention Mechanism)
  * Sorts unseen test designs by uncertainty u(x)
  * Evaluates MAE risk at coverage levels: 100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%
  * Calculates AURC (Area Under Risk-Coverage Curve) vs Random Baseline
  * Demonstrates selective verification speedup & compute savings

Verifies Hypothesis H4:
- Rejecting high-uncertainty designs dramatically lowers retained prediction risk.

Outputs:
- hls_ml/results/calibration_metrics.csv
- hls_ml/results/risk_coverage_data.csv
- hls_ml/results/calibration_selective_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES, COVERAGE_LEVELS, CALIBRATION_LEVELS


def compute_ence(y_true, y_pred, uncertainties, n_bins=10):
    """
    Computes Expected Normalized Calibration Error (ENCE).
    Bins predictions by uncertainty and measures deviation of RMSE from mean predicted uncertainty.
    """
    # Sort by uncertainty
    order = np.argsort(uncertainties)
    bin_size = len(order) // n_bins
    bin_errors = []

    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else len(order)
        idx = order[start:end]

        bin_true = y_true[idx]
        bin_pred = y_pred[idx]
        bin_uncert = uncertainties[idx]

        rmse = np.sqrt(np.mean((bin_true - bin_pred) ** 2))
        mean_sigma = np.mean(bin_uncert)

        if mean_sigma > 1e-6:
            bin_errors.append(np.abs(rmse - mean_sigma) / mean_sigma)
        else:
            bin_errors.append(0.0)

    return float(np.mean(bin_errors))


def compute_aurc(abs_errors, uncertainty, coverages):
    """Compute risk-coverage area after accepting low-uncertainty samples."""
    order = np.argsort(uncertainty)
    risks = []
    for coverage in sorted(coverages):
        count = max(1, int(np.ceil(coverage * len(order))))
        risks.append(float(np.mean(abs_errors[order[:count]])))
    return float(np.trapezoid(risks, sorted(coverages)))


def compute_random_aurc(abs_errors, coverages, repeats=20, seed=2026):
    """Estimate random-rejection AURC over independent permutations."""
    rng = np.random.default_rng(seed)
    values = [compute_aurc(abs_errors, rng.permutation(len(abs_errors)), coverages)
              for _ in range(repeats)]
    return float(np.mean(values)), float(np.std(values))


def main():
    print("=" * 75)
    print("Evaluating Calibration & Risk-Coverage Selective Prediction (Milestone 5)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # [1] Load Predictions & Uncertainty Arrays
    # -------------------------------------------------------------
    npz_path = RESULTS_DIR / "uncertainty_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run train_uncertainty.py first.")

    print("\n[1] Loading uncertainty predictions from Milestone 4...")
    data = np.load(npz_path)

    Y_true = data["Y_test_true"]                      # (39056, 5)
    rf_preds = data["rf_preds"]                        # (39056, 5)
    rf_uncert = data["rf_uncertainty"]                 # (39056, 5)
    rf_pi_lower = data["rf_pi_lower"]                  # (39056, 5) (10th percentile)
    rf_pi_upper = data["rf_pi_upper"]                  # (39056, 5) (90th percentile)
    calibration_residuals = data["rf_calibration_residuals"]

    deep_preds = data["deep_ensemble_preds"]           # (39056, 5)
    deep_uncert = data["deep_ensemble_uncertainty"]    # (39056, 5)

    total_samples = len(Y_true)
    print(f"Loaded {total_samples:,} unseen test samples for 5 targets.")

    # -------------------------------------------------------------
    # [2] E6: Calibration Metrics (PICP, MPIW, ENCE)
    # -------------------------------------------------------------
    print("\n[2] Computing Calibration Metrics (E6)...")
    calib_records = []

    for t_idx, target_name in enumerate(TARGET_NAMES):
        y_t = Y_true[:, t_idx]
        pred_rf = rf_preds[:, t_idx]
        uncert_rf = rf_uncert[:, t_idx]
        low_rf = rf_pi_lower[:, t_idx]
        high_rf = rf_pi_upper[:, t_idx]

        calibration_coverages = {}
        for nominal in CALIBRATION_LEVELS:
            q = np.quantile(calibration_residuals[:, t_idx], nominal)
            lower = np.clip(pred_rf - q, 0, None)
            upper = pred_rf + q
            calibration_coverages[nominal] = float(np.mean((y_t >= lower) & (y_t <= upper)))
        q80 = np.quantile(calibration_residuals[:, t_idx], 0.80)
        low_rf = np.clip(pred_rf - q80, 0, None)
        high_rf = pred_rf + q80
        picp_rf = np.mean((y_t >= low_rf) & (y_t <= high_rf))
        mpiw_rf = np.mean(high_rf - low_rf)
        norm_mpiw_rf = mpiw_rf / (np.median(np.abs(y_t)) + 1e-6)
        ence_rf = compute_ence(y_t, pred_rf, uncert_rf)

        # Deep Ensemble ENCE
        pred_deep = deep_preds[:, t_idx]
        uncert_deep = deep_uncert[:, t_idx]
        ence_deep = compute_ence(y_t, pred_deep, uncert_deep)

        print(f"  [{target_name:<18}] Tree PICP (Nominal 80%): {picp_rf*100:.1f}% | Norm MPIW: {norm_mpiw_rf:.2f} | Tree ENCE: {ence_rf:.3f} | Deep ENCE: {ence_deep:.3f}")

        calib_records.append({
            "Target": target_name,
            "Tree_PICP_80": round(float(picp_rf), 4),
            "Tree_MPIW": round(float(mpiw_rf), 2),
            "Tree_Normalized_MPIW": round(float(norm_mpiw_rf), 4),
            "Tree_ENCE": round(float(ence_rf), 4),
                "Deep_Ensemble_ENCE": round(float(ence_deep), 4),
                **{f"Coverage_{int(level * 100)}": round(value, 4)
                    for level, value in calibration_coverages.items()}
        })

    df_calib = pd.DataFrame(calib_records)
    csv_calib = RESULTS_DIR / "calibration_metrics.csv"
    df_calib.to_csv(csv_calib, index=False)
    print(f"Saved calibration metrics to: {csv_calib}")

    # -------------------------------------------------------------
    # [3] E7: Risk-Coverage Selective Prediction (Abstention)
    # -------------------------------------------------------------
    print("\n[3] Computing Risk-Coverage Curves & Selective Prediction (E7)...")
    rc_records = []

    for method_name, preds_arr, uncerts_arr in [
        ("Tree_Ensemble", rf_preds, rf_uncert),
        ("Deep_Ensemble", deep_preds, deep_uncert)
    ]:
        for t_idx, target_name in enumerate(TARGET_NAMES):
            y_t = Y_true[:, t_idx]
            pred_t = preds_arr[:, t_idx]
            uncert_t = uncerts_arr[:, t_idx]

            abs_errors = np.abs(y_t - pred_t)
            # Sort indices by uncertainty ascending (lowest uncertainty accepted first)
            sorted_indices = np.argsort(uncert_t)

            baseline_mae = float(np.mean(abs_errors))
            aurc = compute_aurc(abs_errors, uncert_t, COVERAGE_LEVELS)
            random_aurc, random_aurc_std = compute_random_aurc(abs_errors, COVERAGE_LEVELS)

            for cov in COVERAGE_LEVELS:
                k = int(np.ceil(cov * total_samples))
                k = max(k, 1)
                accepted_idx = sorted_indices[:k]
                retained_mae = float(np.mean(abs_errors[accepted_idx]))
                retained_ratio = retained_mae / (baseline_mae + 1e-6)

                rc_records.append({
                    "Method": method_name,
                    "Target": target_name,
                    "Coverage": cov,
                    "Accepted_Designs": k,
                    "Retained_MAE": round(retained_mae, 2),
                    "Baseline_100_MAE": round(baseline_mae, 2),
                    "Error_Reduction_Factor": round(1.0 / retained_ratio, 2) if retained_ratio > 0 else 1.0,
                    "Retained_Error_Ratio": round(retained_ratio, 4),
                    "AURC": round(aurc, 4),
                    "Random_AURC_Mean": round(random_aurc, 4),
                    "Random_AURC_Std": round(random_aurc_std, 4)
                })

    df_rc = pd.DataFrame(rc_records)
    csv_rc = RESULTS_DIR / "risk_coverage_data.csv"
    df_rc.to_csv(csv_rc, index=False)
    print(f"Saved Risk-Coverage curve data to: {csv_rc}")

    # -------------------------------------------------------------
    # [4] Generate Executive Summary Report & Scientific Hypothesis Checks
    # -------------------------------------------------------------
    summary_file = RESULTS_DIR / "calibration_selective_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Calibration & Risk-Coverage Selective Prediction Summary (Milestone 5)\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. Hypothesis H4 Verification: Selective Prediction Lowers Error on Accepted Samples\n")
        f.write("-" * 75 + "\n")
        for method in ["Tree_Ensemble", "Deep_Ensemble"]:
            f.write(f"\nMethod: {method}\n")
            for target in TARGET_NAMES:
                sub = df_rc[(df_rc["Method"] == method) & (df_rc["Target"] == target)]
                mae_100 = sub[sub["Coverage"] == 1.0]["Retained_MAE"].values[0]
                mae_80 = sub[sub["Coverage"] == 0.8]["Retained_MAE"].values[0]
                mae_50 = sub[sub["Coverage"] == 0.5]["Retained_MAE"].values[0]
                gain_80 = mae_100 / (mae_80 + 1e-6)
                gain_50 = mae_100 / (mae_50 + 1e-6)
                f.write(f"  * {target:<18}: 100% Cov MAE={mae_100:>10.1f} | 80% Cov MAE={mae_80:>10.1f} ({gain_80:.2f}x better) | 50% Cov MAE={mae_50:>10.1f} ({gain_50:.2f}x better)\n")

        f.write("\n  [VERIFICATION RESULT]:\n")
        h4_values = df_rc[df_rc["Coverage"] == 0.5]["Retained_Error_Ratio"]
        h4_label = "SUPPORTED" if bool((h4_values < 1.0).all()) else "NOT SUPPORTED"
        f.write(f"  - H4 {h4_label}: selective-risk results are reported from the measured data.\n\n")

        f.write("2. Calibration Quality (E6 Summary):\n")
        f.write("-" * 75 + "\n")
        for _, r in df_calib.iterrows():
            f.write(f"  * {r['Target']:<18}: Tree PICP (80% nom) = {r['Tree_PICP_80']*100:.1f}% | ENCE = {r['Tree_ENCE']:.3f} | Deep ENCE = {r['Deep_Ensemble_ENCE']:.3f}\n")

    print(f"\nSaved executive summary report to: {summary_file}")
    print("\n" + "=" * 75)
    print("Milestone 5 completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
