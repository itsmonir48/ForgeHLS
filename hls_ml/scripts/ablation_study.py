"""
ablation_study.py

Executes Feature Ablation Study (Milestone 6 - E8):
- Evaluates Feature Set A (Basic: 3 features)
- Evaluates Feature Set B (Pragma-Aware: 28 features)
- Evaluates Feature Set C (Structural + Pragma: 42 features)

Under E2 (Kernel-Disjoint Leave-Kernel-Out) split across all 5 targets:
Worst-caseLatency, LUT, DSP, FF, BRAM_18K.

Outputs:
- hls_ml/results/feature_ablation_results.csv
- hls_ml/results/ablation_summary.txt
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from config import DATA_DIR, RESULTS_DIR, TARGET_NAMES


def main():
    print("=" * 75)
    print("Running Feature Set Ablation Study (Milestone 6 - E8)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # [1] Load Data, Schema & Splits
    # -------------------------------------------------------------
    print("\n[1] Loading dataset arrays and schema...")
    npz_data = np.load(DATA_DIR / "dataset_arrays.npz")
    splits_data = np.load(DATA_DIR / "splits.npz")

    X_all = npz_data["X_features"]           # (184717, 42)
    Y_all = npz_data["y_targets"]            # (184717, 5)
    all_feature_names = list(npz_data["feature_names"])

    train_idx = splits_data["e2_grouped_train_idx"]
    test_idx = splits_data["e2_grouped_test_idx"]

    # Feature subsets indexing
    idx_set_a = list(range(3))               # 0:3 (3 features)
    idx_set_b = list(range(28))              # 0:28 (28 features)
    idx_set_c = list(range(42))              # 0:42 (42 features)

    feature_sets = [
        ("Set_A_Basic", idx_set_a, 3),
        ("Set_B_PragmaAware", idx_set_b, 28),
        ("Set_C_StructuralPragma", idx_set_c, 42)
    ]

    ablation_records = []

    # -------------------------------------------------------------
    # [2] Train & Evaluate each Feature Set on E2
    # -------------------------------------------------------------
    print("\n[2] Training on E2 (Kernel-Disjoint) with Feature Sets A, B, and C...")

    for set_name, feat_indices, num_feats in feature_sets:
        print(f"\n--- Feature Configuration: {set_name} ({num_feats} features) ---")
        X_sub = X_all[:, feat_indices]

        X_train = X_sub[train_idx]
        X_test = X_sub[test_idx]
        Y_train = Y_all[train_idx]
        Y_test = Y_all[test_idx]

        for t_idx, target_name in enumerate(TARGET_NAMES):
            t0 = time.time()
            y_train_t = Y_train[:, t_idx]
            y_test_t = Y_test[:, t_idx]

            rf = RandomForestRegressor(
                n_estimators=30,
                max_depth=16,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_train, np.log1p(np.clip(y_train_t, 0, None)))

            # Test predictions
            tree_preds = np.expm1(np.array([tree.predict(X_test) for tree in rf.estimators_]))
            tree_preds = np.clip(tree_preds, 0, None)
            pred_mean = np.mean(tree_preds, axis=0)
            uncert_std = np.std(np.log1p(tree_preds), axis=0)

            mae = mean_absolute_error(y_test_t, pred_mean)
            median_ae = median_absolute_error(y_test_t, pred_mean)
            log_mae = mean_absolute_error(np.log1p(y_test_t), np.log1p(pred_mean))
            r2 = r2_score(y_test_t, pred_mean)

            abs_err = np.abs(y_test_t - pred_mean)
            spearman_corr, _ = spearmanr(uncert_std, abs_err)

            elapsed = time.time() - t0
            print(f"  [{target_name:<18}] R²={r2:+.3f} | MAE={mae:>10.1f} | Spearman rho={spearman_corr:.3f} ({elapsed:.1f}s)")

            ablation_records.append({
                "Feature_Set": set_name,
                "Num_Features": num_feats,
                "Target": target_name,
                "R2": round(float(r2), 4),
                "MAE": round(float(mae), 2),
                "MedianAE": round(float(median_ae), 2),
                "LogMAE": round(float(log_mae), 4),
                "Spearman_rho": round(float(spearman_corr), 4),
                "Mean_Uncertainty": round(float(np.mean(uncert_std)), 2)
            })

    # -------------------------------------------------------------
    # [3] Save Results & Summary
    # -------------------------------------------------------------
    df_abl = pd.DataFrame(ablation_records)
    csv_path = RESULTS_DIR / "feature_ablation_results.csv"
    df_abl.to_csv(csv_path, index=False)
    print(f"\n[3] Saved ablation results table to: {csv_path}")

    summary_file = RESULTS_DIR / "ablation_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Feature Set Ablation Study Summary (Milestone 6 - E8)\n")
        f.write("=" * 75 + "\n\n")

        for target in TARGET_NAMES:
            sub = df_abl[df_abl["Target"] == target]
            f.write(f"Target: {target}\n")
            for _, r in sub.iterrows():
                f.write(f"  * {r['Feature_Set']:<24}: MAE={r['MAE']:>12.1f} | R²={r['R2']:+.3f} | Spearman rho={r['Spearman_rho']:.3f}\n")
            f.write("\n")

        f.write("Key Ablation Findings:\n")
        f.write("-" * 75 + "\n")
        f.write("- Feature-set conclusions must be drawn from the per-target metrics above;\n")
        f.write("  no feature set is assumed to improve every target.\n")

    print(f"Saved executive summary report to: {summary_file}")
    print("\n" + "=" * 75)
    print("Ablation study successfully completed!")
    print("=" * 75)


if __name__ == "__main__":
    main()
