"""
test_feature_extractor.py
Unit test and benchmark for feature_extractor.py on actual ForgeHLS records.
"""

import json
import time
from pathlib import Path
from feature_extractor import (
    extract_features_from_record,
    extract_targets_from_record,
    FEATURES_A,
    FEATURES_B,
    FEATURES_C,
    TARGET_COLUMNS
)
from config import RAW_DATASET_PATH as DATA_PATH

def main():
    print("=" * 60)
    print("Testing Feature Extractor (Milestone 1 - Step 1.2 Verification)")
    print("=" * 60)

    print(f"Feature Set A count: {len(FEATURES_A)}")
    print(f"Feature Set B count: {len(FEATURES_B)}")
    print(f"Feature Set C count: {len(FEATURES_C)}")
    print(f"Target count:        {len(TARGET_COLUMNS)}")

    print("\n[1] Reading sample records from dataset...")
    # Load first 500 records
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:500]
    print(f"Loaded {len(samples)} sample records for testing.")

    print("\n[2] Extracting features and targets...")
    start_time = time.time()
    for i, row in enumerate(samples):
        feats = extract_features_from_record(row)
        targets = extract_targets_from_record(row)

        # Assert all keys are present
        for k in FEATURES_C:
            assert k in feats, f"Missing feature {k} in row {i}"
            assert isinstance(feats[k], (int, float)), f"Feature {k} is not numeric in row {i}"
            assert not (feats[k] != feats[k]), f"Feature {k} is NaN in row {i}"

        for t in TARGET_COLUMNS:
            assert t in targets, f"Missing target {t} in row {i}"
            assert isinstance(targets[t], (int, float)), f"Target {t} is not numeric in row {i}"
            assert not (targets[t] != targets[t]), f"Target {t} is NaN in row {i}"

    elapsed = time.time() - start_time
    rate = len(samples) / elapsed
    print(f"Verification SUCCESS! Processed {len(samples)} rows in {elapsed:.3f}s ({rate:.1f} rows/sec).")

    # Print sample feature extraction from first record
    sample_feat = extract_features_from_record(samples[0])
    sample_tgt = extract_targets_from_record(samples[0])
    print("\nSample row extraction:")
    print("  Targets:", sample_tgt)
    print("  Feature Set A:", {k: sample_feat[k] for k in FEATURES_A})
    print("  Sample Pragma Feats (Set B):", {k: sample_feat[k] for k in ["pragma_pipeline_enable", "pragma_unroll_count", "pragma_max_unroll_factor", "pragma_array_partition_cyclic"]})
    print("  Sample Structural Feats (Set C):", {k: sample_feat[k] for k in ["code_line_count", "code_function_count", "code_total_loop_count", "code_max_nesting_depth"]})
    print("\nVerification 100% PASSED!")

if __name__ == "__main__":
    main()
