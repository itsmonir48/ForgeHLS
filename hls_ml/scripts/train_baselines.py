"""
train_baselines.py

Trains and evaluates deterministic baseline QoR predictors across 5 targets:
Targets: Worst-caseLatency, LUT, DSP, FF, BRAM_18K

Evaluates:
- E1: Random Row Split (Optimistic Baseline)
- E2: Kernel-Disjoint Split (Leave-Kernel-Out, Seed 42)
- E3: Repeated Kernel-Disjoint Splits (5 Seeds: 42, 43, 44, 45, 46)

Models:
1. Ridge Regression (L2 Linear)
2. Random Forest Regressor
3. Gradient Boosting (HistGradientBoosting / XGBoost)
4. Multi-Layer Perceptron (MLP)

Verifies Scientific Hypotheses:
- H1: Random row split is optimistic (E1 error << E2 error)
- H2: Kernel-disjoint split introduces distribution shift (E2 error >> E1 error)

Outputs saved to:
- hls_ml/results/baseline_results_e1_vs_e2.csv
- hls_ml/results/baseline_results_e3_repeated.csv
- hls_ml/results/baseline_summary.txt
- hls_ml/results/baseline_e2_predictions.npz (for downstream UQ / calibration)
"""

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

# Suppress expected non-critical convergence and linalg warnings for clean logs
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES, SEEDS


def calculate_metrics(y_true, y_pred):
    """Calculate raw-scale and robust metrics for a non-negative QoR target."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_val = np.mean(np.abs(y_true))
    nmae = mae / (mean_val + 1e-6)  # relative error ratio
    log_mae = mean_absolute_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))
    medae = median_absolute_error(y_true, y_pred)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "NMAE": nmae,
            "MedianAE": medae, "LogMAE": log_mae}


def fit_predict(model_name, X, y, train_idx, test_idx, seed):
    """Fit preprocessing and a log1p target model on one split only."""
    use_scaled = model_name in ["Ridge", "MLP"]
    if use_scaled:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
    else:
        X_train, X_test = X[train_idx], X[test_idx]

    model = get_model(model_name, seed=seed)
    model.fit(X_train, np.log1p(np.clip(y[train_idx], 0, None)))
    return np.clip(np.expm1(model.predict(X_test)), 0, None)


def get_model(model_name, seed=42):
    """Factory for baseline models."""
    if model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    elif model_name == "RandomForest":
        # Fast & strong ensemble
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=16,
            min_samples_split=10,
            n_jobs=-1,
            random_state=seed
        )
    elif model_name == "GradientBoosting":
        # HistGradientBoosting is fast and high-capacity
        return HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=12,
            learning_rate=0.1,
            random_state=seed
        )
    elif model_name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=40,
            batch_size=256,
            early_stopping=True,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    print("=" * 75)
    print("Training Baseline QoR Predictors (Milestone 3: E1, E2, E3)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # [1] Load Data and Splits
    # -------------------------------------------------------------
    print("\n[1] Loading dataset arrays and splits...")
    npz_data = np.load(DATA_DIR / "dataset_arrays.npz")
    splits_data = np.load(DATA_DIR / "splits.npz")

    X = npz_data["X_features"]  # shape (184717, 42)
    Y = npz_data["y_targets"]   # shape (184717, 5)
    feature_names = npz_data["feature_names"]

    print(f"Loaded feature matrix: {X.shape}, targets matrix: {Y.shape}")
    print(f"Features: {len(feature_names)}, Targets: {TARGET_NAMES}")

    model_families = ["Ridge", "RandomForest", "GradientBoosting", "MLP"]
    e1_vs_e2_records = []

    # -------------------------------------------------------------
    # [2] Experiment 1 vs Experiment 2 (Direct H1 & H2 Test)
    # -------------------------------------------------------------
    print("\n[2] Running E1 (Random Row Split) vs E2 (Kernel-Disjoint Split)...")

    # Split indices
    e1_train_idx = splits_data["e1_random_train_idx"]
    e1_test_idx = splits_data["e1_random_test_idx"]
    e2_train_idx = splits_data["e2_grouped_train_idx"]
    e2_test_idx = splits_data["e2_grouped_test_idx"]

    # Store test predictions of E2 for downstream UQ and calibration
    e2_predictions = {}

    for model_name in model_families:
        print(f"\n--- Model Family: {model_name} ---")
        for t_idx, target_name in enumerate(TARGET_NAMES):
            y_curr = Y[:, t_idx]

            # --- E1: Random Split ---
            t0 = time.time()
            pred_e1 = fit_predict(model_name, X, y_curr, e1_train_idx, e1_test_idx, seed=42)
            metrics_e1 = calculate_metrics(y_curr[e1_test_idx], pred_e1)

            # --- E2: Kernel-Disjoint Split ---
            pred_e2 = fit_predict(model_name, X, y_curr, e2_train_idx, e2_test_idx, seed=42)
            metrics_e2 = calculate_metrics(y_curr[e2_test_idx], pred_e2)
            fit_time = time.time() - t0

            # Save prediction for E2
            e2_predictions[f"{model_name}_{target_name}"] = pred_e2

            # Record
            record = {
                "Model": model_name,
                "Target": target_name,
                "E1_R2": round(metrics_e1["R2"], 4),
                "E1_MAE": round(metrics_e1["MAE"], 2),
                "E1_RMSE": round(metrics_e1["RMSE"], 2),
                "E1_NMAE": round(metrics_e1["NMAE"], 4),
                "E1_MedianAE": round(metrics_e1["MedianAE"], 2),
                "E1_LogMAE": round(metrics_e1["LogMAE"], 4),
                "E2_R2": round(metrics_e2["R2"], 4),
                "E2_MAE": round(metrics_e2["MAE"], 2),
                "E2_RMSE": round(metrics_e2["RMSE"], 2),
                "E2_NMAE": round(metrics_e2["NMAE"], 4),
                "E2_MedianAE": round(metrics_e2["MedianAE"], 2),
                "E2_LogMAE": round(metrics_e2["LogMAE"], 4),
                "MAE_Degradation_Ratio": round(metrics_e2["MAE"] / (metrics_e1["MAE"] + 1e-6), 2),
                "Fit_Time_s": round(fit_time, 2)
            }
            e1_vs_e2_records.append(record)

            print(
                f"  [{target_name:<18}] E1(Random) R²={metrics_e1['R2']:.3f}, MAE={metrics_e1['MAE']:.1f} | "
                f"E2(Disjoint) R²={metrics_e2['R2']:.3f}, MAE={metrics_e2['MAE']:.1f} | "
                f"Shift Degradation: {record['MAE_Degradation_Ratio']}x"
            )

    df_e1_e2 = pd.DataFrame(e1_vs_e2_records)
    csv_e1_e2 = RESULTS_DIR / "baseline_results_e1_vs_e2.csv"
    df_e1_e2.to_csv(csv_e1_e2, index=False)
    print(f"\nSaved E1 vs E2 comparative results to: {csv_e1_e2}")

    # Also save E2 predictions for subsequent uncertainty analysis
    e2_predictions["y_true_e2"] = Y[e2_test_idx]
    e2_pred_path = RESULTS_DIR / "baseline_e2_predictions.npz"
    np.savez_compressed(e2_pred_path, **e2_predictions)
    print(f"Saved E2 test predictions to: {e2_pred_path}")

    # -------------------------------------------------------------
    # [3] Experiment 3: Repeated Grouped Splits (5 Seeds Stability)
    # -------------------------------------------------------------
    print("\n[3] Running E3: Repeated Kernel-Disjoint Splits (5 Seeds Stability on Top Models)...")
    # Evaluate best models (RandomForest and GradientBoosting) across all 5 seeds
    e3_records = []

    for model_name in ["RandomForest", "GradientBoosting"]:
        print(f"\nEvaluating stability across 5 seeds for {model_name}...")
        for t_idx, target_name in enumerate(TARGET_NAMES):
            y_curr = Y[:, t_idx]
            seed_r2s = []
            seed_maes = []
            seed_nmaes = []

            for seed in SEEDS:
                train_idx = splits_data[f"e3_grouped_seed_{seed}_train_idx"]
                test_idx = splits_data[f"e3_grouped_seed_{seed}_test_idx"]

                pred = fit_predict(model_name, X, y_curr, train_idx, test_idx, seed=seed)

                m = calculate_metrics(y_curr[test_idx], pred)
                seed_r2s.append(m["R2"])
                seed_maes.append(m["MAE"])
                seed_nmaes.append(m["NMAE"])

            e3_records.append({
                "Model": model_name,
                "Target": target_name,
                "R2_mean": round(float(np.mean(seed_r2s)), 4),
                "R2_std": round(float(np.std(seed_r2s)), 4),
                "MAE_mean": round(float(np.mean(seed_maes)), 2),
                "MAE_std": round(float(np.std(seed_maes)), 2),
                "NMAE_mean": round(float(np.mean(seed_nmaes)), 4),
                "NMAE_std": round(float(np.std(seed_nmaes)), 4),
            })
            print(f"  {target_name:<18}: R² = {np.mean(seed_r2s):.3f} ± {np.std(seed_r2s):.3f} | MAE = {np.mean(seed_maes):.1f} ± {np.std(seed_maes):.1f}")

    df_e3 = pd.DataFrame(e3_records)
    csv_e3 = RESULTS_DIR / "baseline_results_e3_repeated.csv"
    df_e3.to_csv(csv_e3, index=False)
    print(f"\nSaved E3 repeated stability results to: {csv_e3}")

    # -------------------------------------------------------------
    # [4] Generate Executive Summary Report & Scientific Hypothesis Checks
    # -------------------------------------------------------------
    summary_file = RESULTS_DIR / "baseline_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Baseline QoR Prediction & Distribution Shift Summary (E1, E2, E3)\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. Hypothesis H1 & H2 Verification (Random vs Kernel-Disjoint):\n")
        f.write("-" * 75 + "\n")
        for model in model_families:
            sub = df_e1_e2[df_e1_e2["Model"] == model]
            avg_e1_r2 = sub["E1_R2"].mean()
            avg_e2_r2 = sub["E2_R2"].mean()
            avg_deg = sub["MAE_Degradation_Ratio"].mean()
            f.write(f"  * {model:<16}: E1 R² = {avg_e1_r2:.3f}  -->  E2 R² = {avg_e2_r2:.3f} | Average Error Degradation: {avg_deg:.2f}x\n")

        f.write("\n  [VERIFICATION]:\n")
        h1_supported = bool((df_e1_e2["E1_R2"] > df_e1_e2["E2_R2"]).mean() >= 0.75)
        h2_supported = bool((df_e1_e2["E2_MAE"] > df_e1_e2["E1_MAE"]).mean() >= 0.75)
        h1_label = "SUPPORTED" if h1_supported else "NOT SUPPORTED"
        h2_label = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
        f.write(f"  - H1 {h1_label}: the measured random-vs-grouped comparison is reported above.\n")
        f.write(f"  - H2 {h2_label}: the measured random-vs-grouped comparison is reported above.\n\n")

        f.write("2. E3 (Repeated Grouped 5 Seeds Stability - RandomForest):\n")
        f.write("-" * 75 + "\n")
        rf_e3 = df_e3[df_e3["Model"] == "RandomForest"]
        for _, row in rf_e3.iterrows():
            f.write(f"  * {row['Target']:<18}: R² = {row['R2_mean']:.3f} ± {row['R2_std']:.3f} | MAE = {row['MAE_mean']:.1f} ± {row['MAE_std']:.1f}\n")

    print(f"Saved executive summary report to: {summary_file}")
    print("\n" + "=" * 75)
    print("Milestone 3 execution completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
