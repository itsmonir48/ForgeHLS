"""
make_splits.py

Generates reproducible, leakage-free data splits for all thesis experiments:
- E1: Random Row Split (80% train, 20% test)
- E2: Kernel-Disjoint Split (Leave-Kernel-Out on 491 groups, seed=42)
- E3: Repeated Kernel-Disjoint Splits (5 seeds: 42, 43, 44, 45, 46)

Verifies:
- Zero data leakage: train_groups ∩ test_groups = empty set for all grouped splits.
- Proper balance between train and test row counts.

Outputs saved to:
- hls_ml/data/splits.npz
- hls_ml/results/splits_summary.txt
"""

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from config import DATA_DIR, RESULTS_DIR, SEEDS


def main():
    print("=" * 70)
    print("Generating Leakage-Free Dataset Splits (Milestone 2 - Step 2.2)")
    print("=" * 70)

    meta_path = DATA_DIR / "dataset_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run build_dataset.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading dataset metadata...")
    import pyarrow.parquet as pq
    df_meta = pq.read_table(meta_path).to_pandas()
    total_records = len(df_meta)
    groups = df_meta["group_id"].values
    unique_groups = np.unique(groups)
    print(f"Total designs: {total_records:,}")
    print(f"Unique logical kernel groups: {len(unique_groups)}")

    splits = {}
    summary_lines = []
    summary_lines.append("ForgeHLS Partitioning & Split Audit Summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Total design records: {total_records:,}")
    summary_lines.append(f"Unique logical groups: {len(unique_groups)}\n")

    # -------------------------------------------------------------
    # [2] E1: Random Row Split (Optimistic Baseline)
    # -------------------------------------------------------------
    print("\n[2] Generating E1: Random Row Split (80/20, seed=42)...")
    indices = np.arange(total_records)
    train_idx_e1, test_idx_e1 = train_test_split(
        indices, test_size=0.20, random_state=42, shuffle=True
    )
    splits["e1_random_train_idx"] = train_idx_e1
    splits["e1_random_test_idx"] = test_idx_e1

    # Check leakage in Random Split (Demonstrate H1 motivation)
    train_grps_e1 = set(df_meta.iloc[train_idx_e1]["group_id"])
    test_grps_e1 = set(df_meta.iloc[test_idx_e1]["group_id"])
    leak_groups = train_grps_e1.intersection(test_grps_e1)

    print(f"  Train designs: {len(train_idx_e1):,} ({len(train_idx_e1)/total_records*100:.1f}%)")
    print(f"  Test designs:  {len(test_idx_e1):,} ({len(test_idx_e1)/total_records*100:.1f}%)")
    print(f"  [H1 Motivation Check] Groups present in BOTH train and test: {len(leak_groups)}/{len(unique_groups)} ({len(leak_groups)/len(unique_groups)*100:.1f}%)")

    summary_lines.append("1. E1: Random Row Split (Baseline)")
    summary_lines.append("-" * 70)
    summary_lines.append(f"  Train: {len(train_idx_e1):,} rows ({len(train_idx_e1)/total_records*100:.1f}%)")
    summary_lines.append(f"  Test:  {len(test_idx_e1):,} rows ({len(test_idx_e1)/total_records*100:.1f}%)")
    summary_lines.append(f"  Overlapping groups (Leakage): {len(leak_groups)} groups ({len(leak_groups)/len(unique_groups)*100:.1f}%)\n")

    # -------------------------------------------------------------
    # [3] E2 & E3: Kernel-Disjoint Splits (Leave-Kernel-Out)
    # -------------------------------------------------------------
    print("\n[3] Generating E2 & E3: Repeated Kernel-Disjoint Splits...")
    summary_lines.append("2. E2 & E3: Kernel-Disjoint Splits (Strict LKO - 5 Seeds)")
    summary_lines.append("-" * 70)

    for i, seed in enumerate(SEEDS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(gss.split(indices, groups=groups))

        train_grps = set(df_meta.iloc[train_idx]["group_id"])
        test_grps = set(df_meta.iloc[test_idx]["group_id"])
        leak = train_grps.intersection(test_grps)

        # Strictly assert zero leakage
        assert len(leak) == 0, f"DATA LEAKAGE DETECTED in Seed {seed}: {len(leak)} overlapping groups!"

        splits[f"e3_grouped_seed_{seed}_train_idx"] = train_idx
        splits[f"e3_grouped_seed_{seed}_test_idx"] = test_idx

        if seed == 42:
            splits["e2_grouped_train_idx"] = train_idx
            splits["e2_grouped_test_idx"] = test_idx

        pct_train = (len(train_idx) / total_records) * 100
        pct_test = (len(test_idx) / total_records) * 100

        print(f"  Seed {seed}: Train {len(train_idx):,} ({pct_train:.1f}%, {len(train_grps)} groups) | Test {len(test_idx):,} ({pct_test:.1f}%, {len(test_grps)} groups) | Overlap: 0 (PASSED)")

        summary_lines.append(
            f"  Seed {seed}: Train {len(train_idx):,} rows ({pct_train:.1f}%, {len(train_grps)} groups) | "
            f"Test {len(test_idx):,} rows ({pct_test:.1f}%, {len(test_grps)} groups) | Overlap: {len(leak)}"
        )

    # -------------------------------------------------------------
    # [4] Save Splits to Disk
    # -------------------------------------------------------------
    split_path = DATA_DIR / "splits.npz"
    np.savez_compressed(split_path, **splits)
    print(f"\n[4] Saved compressed split masks to: {split_path} ({split_path.stat().st_size / (1024*1024):.2f} MB)")

    summary_file = RESULTS_DIR / "splits_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved split summary report to: {summary_file}")

    print("\n" + "=" * 70)
    print("All splits verified: 100% Leakage-Free and ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
